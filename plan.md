# TWII Next-Day Return Forecasting — Build Plan

## 0. Goal & honest scope

Train **a single LightGBM regressor** to predict the **next-day log return** of 台灣加權指數 (TAIEX, `^TWII`) from technical price/volume indicators. The deliverable is **predictive power**, not a trading book: MAE / RMSE on returns, directional accuracy, and information coefficient — each reported **against a naive baseline** so the numbers mean something. No long/short construction.

The realistic expectation, stated up front so nobody is fooled: out-of-sample $R^2$ on daily returns will be near zero or negative, and directional accuracy in the **51–54%** range is a _good_ result. Anything dramatically higher on a single test split is a bug or leakage, not alpha.

---

## 1. Environment (uv)

`pyproject.toml` dependencies block:

```toml
dependencies = [
    "evidently>=0.7.21",
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
│   ├── target.py            # next-day log return
│   ├── dataset.py           # assemble X (features+lags) and y, drop warmup NaNs
│   ├── split.py             # chronological 85/5/10
│   ├── scaling.py           # quantile scaler, FIT ON TRAIN ONLY
│   ├── model.py             # LightGBM train + early stopping + tuning
│   ├── evaluate.py          # metrics + baselines
│   ├── monitor.py           # evidently train-vs-test drift (MLOps touch)
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
- History: use **~2015 → present**. Full history reaches 1997 but the Asian-crisis-era regime adds noise; the most-recent-10% test window is modern regardless.

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

**Drop** _Cumulative Daily Return_ (non-stationary, trends with the index and breaks across regimes).

**Leakage guard:** every rolling op must be **trailing** (`min_periods=window`, never centered). Compute features on the full series _before_ splitting — this is safe because each value at $t$ depends only on $\le t$. `tests/test_leakage.py` should assert that shifting any feature forward in time changes the test-set predictions (i.e. no future bleed).

After feature generation, **drop the first ~60 warmup rows** (NaNs from the longest window).

---

## 6. Split (`split.py`) — chronological, contiguous

Oldest **85%** → train, next **5%** → validation, most recent **10%** → test. No shuffling, ever.

```python
n = len(df)
i_tr, i_va = int(0.85 * n), int(0.90 * n)
train, val, test = df[:i_tr], df[i_tr:i_va], df[i_va:]
```

---

## 7. Scaling (`scaling.py`) — quantile, fit on train only

RobustScaler-style, fit **only on train**, applied to val/test (the only leakage-free option):

$$x' = \frac{x - Q_{25}^{\text{train}}(x)}{Q_{75}^{\text{train}}(x) - Q_{25}^{\text{train}}(x)}$$

Use `sklearn.preprocessing.RobustScaler(quantile_range=(25, 75))`. Strictly speaking LightGBM doesn't need this (trees split on order, not magnitude) — apply it anyway per spec, and so the feature matrix is model-agnostic if the model is ever swapped. **Scale features only, not the target.**

---

## 8. Model (`model.py`) — single LightGBM

`LGBMRegressor` with early stopping on the validation set.

- **Objective:** `regression_l1` (MAE) or `huber`. Daily returns are **leptokurtic / fat-tailed**, so an L1 or Huber loss is more robust to tail days than L2 — a deliberate, defensible choice. Report both MAE and RMSE regardless.
- **Early stopping:** large `n_estimators` (e.g. 2000) with `early_stopping_rounds=50` monitoring val loss.
- **Tuning:** small **random search** (≈30–50 draws, `tqdm`) over `num_leaves`, `learning_rate`, `max_depth`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`. Select the config with **lowest validation MAE**. Single split (no walk-forward / CPCV — the agreed simplification).
- Seed everything from `config.SEED` for repeatability.

Persist the fitted model and the chosen hyperparameters.

---

## 9. Evaluation (`evaluate.py`) — the actual deliverable

On the **test** split, in return units:

$$\text{MAE} = \tfrac{1}{N}\sum|r_{t+1}-\hat r_{t+1}|, \quad \text{RMSE} = \sqrt{\tfrac{1}{N}\sum (r_{t+1}-\hat r_{t+1})^2}$$

$$\text{DirAcc} = \tfrac{1}{N}\sum \mathbb{1}\!\left[\text{sign}(\hat r_{t+1}) = \text{sign}(r_{t+1})\right], \qquad \text{IC} = \text{corr}(\hat r, r)$$

Plus out-of-sample $R^2$ (expect ~0 or negative — report it honestly).

**Baselines (this is what makes it not theater):**

1. **Persistence:** $\hat r_{t+1} = 0$ → MAE $=\overline{|r|}$, RMSE $=\sqrt{\overline{r^2}}$. The number to beat.
2. **Historical mean:** $\hat r_{t+1} = \bar r_{\text{train}}$ (constant).

Print a table: model vs. both baselines on every metric. If LightGBM doesn't beat persistence on MAE/RMSE and 50% on direction, that's the honest finding — and far more credible than an inflated price-level $R^2$.

---

## 10. Monitoring (`monitor.py`) — MLOps touch

Use **evidently** to generate a train-vs-test **data drift** report on the feature matrix. Directly relevant to the MLOps pillar, and it surfaces whether the test regime differs from train (it usually does for markets) — context for interpreting the test metrics.

---

## 11. Plots (`plots.py`)

- `mplfinance` candlestick of the test window.
- Predicted vs. actual returns scatter (with IC annotated).
- Residual histogram (eyeball the fat tails).
- _Optional, illustrative only:_ reconstruct a price path by compounding $\hat r_{t+1}$ and overlay on actual — not a metric, just a sanity visual.

---

## 12. Build order for the agent

1. `config.py` → `data.py` (+ volume gate) → cache parquet.
2. `features.py` → `target.py` → `dataset.py` → `tests/test_leakage.py` **must pass before proceeding**.
3. `split.py` → `scaling.py` (fit-on-train).
4. `model.py`: baseline LightGBM → random-search tune on val → refit.
5. `evaluate.py`: metrics + baseline table.
6. `monitor.py`, `plots.py`.
7. `run_pipeline.ipynb`: orchestrate, render, write results.

The two non-negotiable correctness gates: **`test_leakage.py` passes**, and **every metric is reported next to its baseline**. Everything else is mechanical.
