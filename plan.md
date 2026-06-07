# TWII Next-Day Return Forecasting — Build Plan

## 0. Goal & honest scope

Train **a single LightGBM regressor** to predict the **next-day log return** of 台灣加權指數 (TAIEX, `^TWII`) from technical price/volume indicators. The deliverable is **predictive power**, not a trading book: MAE / RMSE on returns, directional accuracy, and information coefficient — each reported **against a naive baseline** so the numbers mean something. No long/short construction.

The realistic expectation, stated up front so nobody is fooled: out-of-sample $R^2$ on daily returns will be near zero or negative, and directional accuracy in the **51–54%** range is a _good_ result. Anything dramatically higher on a single test split is a bug or leakage, not alpha.

---

## 1. Environment (uv)

`pyproject.toml` dependencies block:

```toml
dependencies = [
    "lightgbm>=4.6.0",
    "matplotlib>=3.10.9",
    "mplfinance>=0.12.10b0",
    "numpy>=2.4.6",
    "pandas>=3.0.3",
    "pyarrow>=24.0.0",
    "scikit-learn>=1.9.0",
    "tqdm>=4.68.1",
    "yfinance>=1.4.1",
    "jupyter>=1.1.1",
    "nbformat>=5.10.4",
    "ipympl>=0.10.0",
    "pytest>=8.0.0",
    "shap>=0.52.0",
]
```

```bash
uv sync
```

---

## 2. Repository layout

```
twii-forecast/
├── pyproject.toml
├── plan.md
├── data/
│   ├── raw/                 # cached yfinance pull (parquet)
│   └── processed/           # feature matrix (parquet)
├── src/twii_forecast/
│   ├── config.py            # all constants: windows, split ratios, paths, seed
│   ├── data.py              # download + data-quality gate
│   ├── features.py          # technical indicators (causal/trailing only)
│   ├── feature_selection.py # temporal consistency analysis (AUC/ROC)
│   ├── target.py            # next-day log return
│   ├── dataset.py           # assemble X (features+lags) and y, drop warmup NaNs
│   ├── split.py             # chronological 85/5/10
│   ├── scaling.py           # quantile scaler, FIT ON TRAIN ONLY
│   ├── model.py             # LightGBM train + early stopping + tuning
│   ├── evaluate.py          # metrics + baselines
│   ├── explain.py           # SHAP feature importance
│   └── plots.py             # mplfinance candles, pred-vs-actual, residuals
├── notebooks/
│   └── run_pipeline.ipynb   # orchestrates the above end-to-end
└── tests/
    └── test_leakage.py      # asserts no future info in features
```

Keep logic in `src/`, keep `run_pipeline.ipynb` thin — it just calls functions and renders plots. That's the clean separation the old codebase lacked.

---

## 3. Data (`data.py`) — and the volume question

Pull `^TWII` daily OHLCV via `yfinance`, cache to `data/raw/twii.parquet`.

**The caveat:** TAIEX volume is conceptually real (total matched turnover across all listed names). But Yahoo's `^TWII` feed frequently returns **Volume = 0 or NaN** for an index ticker, because Yahoo doesn't reliably populate index turnover. So the plan includes a **data-quality gate**, not an assumption:

```python
def validate_volume(df) -> bool:
    frac_zero = (df["Volume"] == 0).mean()
    frac_nan  = df["Volume"].isna().mean()
    return (frac_zero + frac_nan) < 0.05   # usable if <5% degenerate
```

- If the gate **passes** → keep all volume features ($\Delta V$, $\Delta$Volume MA$_n$).
- If it **fails** → fetch authoritative TAIEX turnover (成交金額/成交量) from the **TWSE OpenAPI** (`openapi.twse.com.tw`, daily market statistics endpoint) and join it on date, _or_ drop volume features with a logged warning. Implement the gate and pick the branch at runtime — do not hard-code an assumption.

Other data notes:

- `^TWII` has **no separate Adjusted Close**. An index has no dividend reinvestment adjustment, so set $AC := C$. This collapses all AC-based indicators onto their Close equivalents — **drop the duplicates** rather than feed perfectly collinear columns.
- History: use **~2005 → present**. Full history reaches 1997 but the Asian-crisis-era regime adds noise; the most-recent-10% test window is modern regardless.

---

## 4. Target (`target.py`)

Predict the next-day log return:

$$r_{t+1} = \ln\!\left(\frac{C_{t+1}}{C_t}\right)$$

This is stationary and the real test. The model is

$$\hat{r}_{t+1} = f\big(\,x_{t-k:t}\,\big), \qquad k = 60$$

where $x$ are technical indicators evaluated **at or before** $t$. **Leave the target unscaled** so MAE/RMSE come out in interpretable return units.

---

## 5. Features (`features.py`) — the $k=60$ window, encoded causally

The 60-day lookback is carried **by the indicators themselves** (MA/EMA up to 60d, RSI-14, BB-21, MACD all summarize trailing history) plus a small set of explicit lagged returns. **Do not** build a raw $60 \times n$ flattened matrix — that floods a single tree model with noise.

From `technical-indicator.md`, with $AC := C$ and duplicates removed:

| Feature                                               | Definition                                              |
| ----------------------------------------------------- | ------------------------------------------------------- |
| Amplitude                                             | $(H_t - L_t)/C_{t-1}$                                   |
| Difference                                            | $(C_t - O_t)/C_{t-1}$ ← _paper's top SHAP feature_      |
| H−L, C−O                                              | raw spreads                                             |
| $\Delta C$, $\Delta V$                                | first differences                                       |
| Intraday MA$_n$, $\Delta C$ MA$_n$, $\Delta V$ MA$_n$ | $n \in \{5,10,20\}$                                     |
| $\pm\Delta$Open                                       | $\text{sign}(O_t - O_{t-1})$                            |
| Daily Return, Momentum                                | $r_t$, $C_t - C_{t-10}$                                 |
| RSI                                                   | 14-day                                                  |
| Williams %R                                           | $(H_{\max}-C_t)/(H_{\max}-L_{\min})\cdot(-100)$, 14-day |
| MA$_n$, EMA$_n$                                       | $n \in \{5,10,20,60\}$                                  |
| MACD                                                  | $\text{EMA}_{12} - \text{EMA}_{26}$                     |
| BB High / Low                                         | $\bar C_{21} \pm 2\,\sigma_{C,21}$                      |
| Lagged returns                                        | $r_t, r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}$               |

**Drop** _Cumulative Daily Return_ (non-stationary, trends with the index and breaks across regimes) — or replace it with a windowed version if you want it.

**Leakage guard:** every rolling op must be **trailing** (`min_periods=window`, never centered). Compute features on the full series _before_ splitting — this is safe because each value at $t$ depends only on $\le t$. `tests/test_leakage.py` should assert that shifting any feature forward in time changes the test-set predictions (i.e. no future bleed).

After feature generation, **drop the first ~60 warmup rows** (NaNs from the longest window).

---

## 6. Feature selection (`feature_selection.py`) — Temporal Consistency Analysis

Before training, weed out features whose distribution **drifts across time**, because a model trained on the past won't generalize on a non-stationary column. The paper does this with the **AUC of a ROC curve**, used in an inverted, non-standard way: AUC measures how easily a binary classifier can tell _which time period_ a sample came from, using only that one feature.

Reading the AUC (per the paper, ranges $[0, 1]$):

| AUC           | Meaning for a feature                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| $1.0$         | perfectly separable — the feature alone tells the two periods apart (severe drift)                                       |
| $0.9{-}1.0$   | outstanding separation → heavy drift                                                                                     |
| $0.7{-}0.9$   | strong separation → drift                                                                                                |
| $\approx 0.5$ | TPR = FPR, the diagonal $y=x$ — periods are **indistinguishable** → the feature is **stable / stationary** → **keep it** |
| $0.0$         | perfectly (inversely) separable → also drift                                                                             |

**The target is AUC $\approx 0.5$** — that means "no drift," which is the desirable outcome here. This is the same idea as drift detection in `monitor.py`, hand-rolled as a feature filter.

Procedure (matching the paper):

1. **Trimonthly blocks.** A monthly split gives only ~20 business days per feature — too few to be statistically meaningful. So split the history into consecutive **3-month** periods instead.
2. **Pairwise permutation.** For each feature, take every **non-overlapping pair** of 3-month blocks, label one block class 0 and the other class 1, and compute the AUC for separating them using that single feature (`sklearn.metrics.roc_auc_score`). A trivial 1-feature classifier — just rank by the feature value — suffices; no model needs to be fit.
3. **Aggregate.** Average (or take the max of) the pairwise AUCs per feature. A feature whose aggregated AUC sits far from $0.5$ is temporally inconsistent → **drop it**.
4. **Threshold.** Keep features with aggregated AUC within a band around $0.5$ (e.g. $\le 0.7$, since the paper treats $[0.7, 0.9]$ as "strong separation"). Make the band a `config` constant and **log the surviving feature set**.

Fit this **on the training period only** (selection is part of model building; the test window must not influence which features are kept).

---

## 7. Split (`split.py`) — chronological, contiguous

Oldest **85%** → train, next **5%** → validation, most recent **10%** → test. No shuffling, ever.

```python
n = len(df)
i_tr, i_va = int(0.85 * n), int(0.90 * n)
train, val, test = df[:i_tr], df[i_tr:i_va], df[i_va:]
```

---

## 8. Scaling (`scaling.py`) — quantile, fit on train only

RobustScaler-style, fit **only on train**, applied to val/test (the only leakage-free option):

$$x' = \frac{x - Q_{25}^{\text{train}}(x)}{Q_{75}^{\text{train}}(x) - Q_{25}^{\text{train}}(x)}$$

Use `sklearn.preprocessing.RobustScaler(quantile_range=(25, 75))`. Strictly speaking LightGBM doesn't need this (trees split on order, not magnitude) — apply it anyway per spec, and so the feature matrix is model-agnostic if the model is ever swapped. **Scale features only, not the target.**

---

## 9. Model (`model.py`) — single LightGBM

`LGBMRegressor` with early stopping on the validation set.

- **Objective:** `regression_l1` (MAE) or `huber`. Daily returns are **leptokurtic / fat-tailed**, so an L1 or Huber loss is more robust to tail days than L2 — a deliberate, defensible choice. Report both MAE and RMSE regardless.
- **Early stopping:** large `n_estimators` (e.g. 2000) with `early_stopping_rounds=50` monitoring val loss.
- **Tuning:** small **random search** (≈30–50 draws, `tqdm`) over `num_leaves`, `learning_rate`, `max_depth`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`. Select the config with **lowest validation MAE**. Single split (no walk-forward / CPCV — the agreed simplification).
- Seed everything from `config.SEED` for repeatability.

Persist the fitted model and the chosen hyperparameters.

---

## 10. Evaluation (`evaluate.py`) — the actual deliverable

**The headline result is next-day prediction accuracy, evaluated per-prediction — never a compounded prediction level.** Every metric below is computed directly on the set of one-step-ahead predictions $\{(\hat r_{t+1}, r_{t+1})\}$ over the test window. Do **not** chain predictions into a multi-step forecast and do **not** compound $\hat r$ into a reconstructed price series for scoring — compounding accumulates error, hides per-day accuracy, and re-introduces the autocorrelated-level trap. The model predicts one day forward from _actual_ observed inputs at each $t$; that single-step accuracy is the deliverable.

On the **test** split, in return units:

$$\text{MAE} = \tfrac{1}{N}\sum|r_{t+1}-\hat r_{t+1}|, \quad \text{RMSE} = \sqrt{\tfrac{1}{N}\sum (r_{t+1}-\hat r_{t+1})^2}$$

$$\text{DirAcc} = \tfrac{1}{N}\sum \mathbb{1}\!\left[\text{sign}(\hat r_{t+1}) = \text{sign}(r_{t+1})\right], \qquad \text{IC} = \text{corr}(\hat r, r)$$

Plus out-of-sample $R^2$ (expect ~0 or negative — report it honestly). **Directional accuracy is the headline number** — the clearest statement of "did the model call tomorrow's move correctly."

**Baselines (this is what makes it not theater):**

1. **Persistence:** $\hat r_{t+1} = 0$ → MAE $=\overline{|r|}$, RMSE $=\sqrt{\overline{r^2}}$. The number to beat.
2. **Historical mean:** $\hat r_{t+1} = \bar r_{\text{train}}$ (constant).

Print a table: model vs. both baselines on every metric. If LightGBM doesn't beat persistence on MAE/RMSE and 50% on direction, that's the honest finding — and far more credible than an inflated price-level $R^2$.

---

## 11. Feature importance (`explain.py`) — SHAP

Explain the trained LightGBM with the **SHAP** framework (Shapley Additive exPlanations), a unified game-theory approach that introduces additive feature-importance measures — addressing the accuracy-vs-interpretability tension that makes ensemble/tree models hard to read. Use the official `shap` Python package.

- Use `shap.TreeExplainer` (exact and fast for LightGBM) on the trained model over the test (or a representative) set.
- **Beeswarm summary plot** — global importance, each feature's distribution of SHAP values and its direction of effect.
- **Bar plot** of mean $|\text{SHAP}|$ — ranked global importance.
- _Optional:_ a few **waterfall/force plots** for individual test-day predictions to show how features push a single forecast above/below the base value.

Sanity check against the paper: it found `Difference` and `C-O` (both derived from $C_t - O_t$) dominated importance. Worth seeing whether the same holds for next-day **return** prediction on TWII, or whether the ranking shifts.

---

## 13. Plots (`plots.py`)

- `mplfinance` candlestick of the test window.
- Predicted vs. actual returns scatter (with IC annotated) — the per-prediction view.
- Residual histogram (eyeball the fat tails).
- **Not a result, sanity-only:** a compounded price path from $\hat r_{t+1}$ overlaid on actual may be drawn purely as a visual gut-check. It is explicitly **excluded from the reported metrics** (§10) — accuracy is reported per next-day prediction, not on any compounded level.

---

## 14. Build order for the agent

1. `config.py` → `data.py` (+ volume gate) → cache parquet.
2. `features.py` → `target.py` → `dataset.py` → `tests/test_leakage.py` **must pass before proceeding**.
3. `feature_selection.py`: temporal consistency (AUC ≈ 0.5) on the train period → log surviving features.
4. `split.py` → `scaling.py` (fit-on-train).
5. `model.py`: baseline LightGBM → random-search tune on val → refit.
6. `evaluate.py`: per-next-day metrics + baseline table (no compounding).
7. `explain.py`: SHAP importance on the trained model.
8. `monitor.py`, `plots.py`.
9. `run_pipeline.ipynb`: orchestrate, render, write results.

The non-negotiable correctness gates: **`test_leakage.py` passes**, **feature selection runs on train only**, **every metric is reported next to its baseline**, and **results are per next-day prediction, never a compounded level**. Everything else is mechanical.

---

## Appendix A — SHAP mathematical detail (for the coding agent)

> The agent does **not** implement any of this; `shap.TreeExplainer` computes it. This appendix is so the agent understands what the numbers mean and can interpret/plot them correctly.

**1. Shapley value (cooperative game theory).** Treat each feature as a "player" and the model's output as the payout to be divided. For feature $i$ out of $M$ features $N = \{1, \dots, M\}$, its Shapley value is the average marginal contribution over all subsets $S$ that exclude $i$:

$$\phi_i = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!\,(M-|S|-1)!}{M!}\,\big[v(S\cup\{i\}) - v(S)\big]$$

The combinatorial weight is the fraction of feature orderings in which exactly the members of $S$ precede $i$. Equivalently, average the marginal contribution over all $M!$ orderings of the features.

**2. The coalition value $v(S)$.** For SHAP, the "payout when only the features in $S$ are known" is the model's expected output with the remaining features marginalized out:

$$v(S) = \mathbb{E}\big[f(x) \mid x_S\big]$$

**3. Additive feature-attribution (explanation) model.** SHAP explains one prediction with a linear model over binary present/absent indicators $z' \in \{0,1\}^M$:

$$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i\,z'_i, \qquad \phi_0 = \mathbb{E}[f(x)]$$

$\phi_0$ is the **base value** (the average prediction over the background data).

**4. Local accuracy (the paper's Eq. 2).** With all features present ($z' = \mathbf{1}$), the explanation must equal the actual prediction:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

So the SHAP values **exactly decompose the gap** between this prediction and the average prediction. Each $\phi_i$ is how much feature $i$ pushed this specific prediction above (+) or below (−) the base value.

**5. Uniqueness.** These three properties pin $\phi_i$ down uniquely:

- **Local accuracy / efficiency** — attributions sum to $f(x) - \phi_0$ (property 4 above).
- **Missingness** — a feature absent from the input gets $\phi_i = 0$.
- **Consistency** — if a model changes so a feature's marginal contribution rises (or stays equal) in every subset, its attribution cannot fall.

**6. Tractability.** The exact sum runs over $2^M$ subsets (exponential). For **tree models**, **TreeSHAP** computes the exact values in polynomial time, $O(TLD^2)$ for $T$ trees, $L$ leaves, depth $D$ — this is what `shap.TreeExplainer` uses on LightGBM.

**7. Global importance from local values.** Per-prediction $\phi_i$ are aggregated across the test set:

- mean $|\text{SHAP}|$: $\;\bar I_i = \frac{1}{N}\sum_{n=1}^{N} |\phi_i^{(n)}|\;$ → the ranked bar plot.
- the full distribution of $\phi_i^{(n)}$ colored by feature value → the beeswarm summary plot.

**Plotting calls (no math needed):**

```python
import shap
explainer = shap.TreeExplainer(model)
sv = explainer(X_test)          # SHAP values for the test matrix
shap.plots.beeswarm(sv)         # global importance + direction
shap.plots.bar(sv)              # ranked mean |SHAP|
shap.plots.waterfall(sv[0])     # one prediction's decomposition
```

---

## Appendix B — Temporal Consistency Analysis, formalized (for the coding agent)

Implements §6. Goal: drop features whose distribution **drifts across time**, because a model trained on the past won't generalize on a non-stationary column. Drift is measured by how well a single feature can separate one time period from another, scored by **ROC AUC**.

**1. ROC / AUC recap.** A binary classifier outputs scores; sweeping a decision threshold traces the ROC curve of $(\text{FPR}, \text{TPR})$:

$$\text{TPR} = \frac{TP}{TP+FN}, \qquad \text{FPR} = \frac{FP}{FP+TN}$$

$\text{AUC} = \int_0^1 \text{TPR}\,d(\text{FPR}) \in [0,1]$. Interpretation (from the paper):

| AUC           | Meaning                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------- |
| $1.0$         | perfectly separable (severe drift)                                                                 |
| $0.9{-}1.0$   | outstanding separation → heavy drift                                                               |
| $0.7{-}0.9$   | strong separation → drift                                                                          |
| $\approx 0.5$ | $\text{TPR} = \text{FPR}$, the diagonal $y = x$ — periods **indistinguishable** → **stable, keep** |
| $0.0$         | perfectly inversely separable → drift                                                              |

AUC also equals the probability the classifier ranks a random positive above a random negative, which is why a **single feature** suffices as the "classifier": just rank samples by that feature's value and let `sklearn.metrics.roc_auc_score(period_label, feature_values)` score the separation. No model is fitted.

**2. Inverted target.** Here, **AUC $\approx 0.5$ is the _good_ outcome** — it means the feature looks the same across periods (no drift). This is drift detection used as a feature filter.

**3. Procedure.**

1. **Trimonthly blocks.** ~20 business days/month is too few to compare months robustly, so split the (training) history into consecutive **3-month** blocks $B_1, B_2, \dots, B_P$.
2. **Pairwise permutation.** For each feature $j$ and each **non-overlapping pair** of blocks $(B_a, B_b)$, $a < b$: label $B_a$ as class $0$, $B_b$ as class $1$, and compute
   $$\text{AUC}_{j}^{(a,b)} = \text{roc\_auc\_score}\big(y_{ab},\; x_{j,\,ab}\big)$$
   where $x_{j,ab}$ is feature $j$'s values over the two blocks. (Use $\max(\text{AUC}, 1-\text{AUC})$ so direction of separation doesn't matter — both extremes are drift.)
3. **Aggregate per feature.** $\;\bar A_j = \operatorname{agg}_{a<b}\,\text{AUC}_j^{(a,b)}\;$ (mean or max; make it a `config` choice).
4. **Threshold.** Keep feature $j$ iff $\bar A_j \le \tau$ (the paper's band sits near $0.5$; treat $[0.7,0.9]$ as "strong separation," so e.g. $\tau = 0.7$). Expose $\tau$ in `config` and **log the surviving feature set**.

**4. Leakage rule.** Fit selection on the **training period only**; the val/test windows must never influence which features survive.

```python
from itertools import combinations
import numpy as np
from sklearn.metrics import roc_auc_score

def temporal_consistency(df_train, feature_cols, tau=0.7, agg="mean"):
    blocks = make_trimonthly_blocks(df_train)        # list of index arrays
    keep = []
    for j in feature_cols:
        scores = []
        for a, b in combinations(range(len(blocks)), 2):
            x = np.r_[df_train.loc[blocks[a], j].values,
                      df_train.loc[blocks[b], j].values]
            y = np.r_[np.zeros(len(blocks[a])), np.ones(len(blocks[b]))]
            auc = roc_auc_score(y, x)
            scores.append(max(auc, 1 - auc))         # direction-agnostic
        agg_auc = np.mean(scores) if agg == "mean" else np.max(scores)
        if agg_auc <= tau:
            keep.append(j)
    return keep
```
