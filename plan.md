# Quant Demo Project — Build Plan

A self-contained, laptop-scale machine learning quant research loop, built to **demonstrate methodological rigor** for a second-round ML Research Engineer interview. The goal is _not_ to find deployable alpha — it is to prove you can stand up the full pipeline (data → alpha → model → backtest → portfolio → execution → monitoring) and stress-test it honestly. The target universe is the **top 50 companies on the Taiwan Stock Exchange (TWSE)** by market cap / liquidity.

> **Guiding principle:** A modest, honest result with airtight methodology beats a Sharpe-of-3 fantasy. Every stage should make a senior quant think "this person has done it before."

---

## 0. Scope & success criteria

**In scope**

- Cross-sectional, dollar-neutral, long-short equity strategy on daily data over the **TWSE top 50** universe.
- Full reproducible pipeline with caching, leakage controls, cost modeling, and drift monitoring.

> **Universe-size caveat (say this out loud):** 50 names is a small cross-section. Decile long-short would leave ~5 names per leg — too noisy. Use **quintiles** (top/bottom 10 names) instead, and lean on IC as the primary signal-quality metric since portfolio PnL is high-variance with so few names.

**Out of scope (name these as future work, don't try to build them)**

- High-frequency / intraday data, order-flow features, point-in-time survivorship-free universe, live execution.

**Definition of done**

- [ ] `python run_pipeline.py` reproduces every number and chart from a cold cache.
- [ ] Reported metrics: mean IC, ICIR, gross & **net** Sharpe, max drawdown, turnover.
- [ ] Purged + embargoed walk-forward validation is the _only_ evaluation used.
- [ ] An Evidently drift report renders as an HTML artifact.
- [ ] You can defend every design choice in one sentence (see §7).

**Target numbers (sanity band — if you blow past these, suspect a bug)**

- Mean daily IC ≈ 0.01–0.04, ICIR ≈ 0.5–2.0 (annualized).
- Gross Sharpe ≈ 0.8–1.8; **net Sharpe ≈ 0.3–1.0**. A net Sharpe > 2 on free daily data = leakage. Go hunting.

---

## 1. Environment & repo setup (≈ 30 min)

### 1.1 Environment

```bash
source ./.venv/bin/activate
uv add pandas numpy yfinance lightgbm scikit-learn scipy matplotlib pyarrow evidently tqdm
```

> **Data note:** TWSE tickers on Yahoo Finance use the `.TW` suffix (e.g. `2330.TW` for TSMC). The market index proxy is `^TWII` (TAIEX); the liquid ETF proxy is `0050.TW` (Yuanta Taiwan Top 50, which itself tracks this universe).

When you try to run any Python code, use

```bash
uv run <script>.py
```

### 1.2 Repo structure

```
quant-demo/
├── README.md                 # 1-paragraph framing + how to run
├── requirements.txt
├── config.py                 # universe, dates, horizon, costs, CV params
├── run_pipeline.py           # orchestrates the whole loop end-to-end
├── data/
│   ├── ingest.py             # yfinance pull + parquet cache
│   └── cache/                # raw parquet (git-ignored)
├── features/
│   └── alpha.py              # per-stock features + cross-sectional ranking
├── model/
│   ├── validation.py         # purged, embargoed walk-forward splitter
│   └── train.py              # LightGBM train/predict over walk-forward
├── backtest/
│   ├── metrics.py            # IC, ICIR
│   ├── portfolio.py          # quintile long-short, sector-neutral, weights
│   └── engine.py             # vectorized backtester + transaction costs
├── monitoring/
│   └── drift.py              # rolling IC, PSI/KS, Evidently report
├── reports/                  # generated charts + drift html (git-ignored)
└── notebooks/
    └── explore.ipynb         # scratch only — production logic lives in modules
```

> **Discipline that scores points:** production logic in modules, notebooks for exploration only. Mention this — it signals you write code others can run.

### 1.3 `config.py`

```python
START, END        = "2014-01-01", "2024-12-31"
UNIVERSE          = "TWSE_TOP50"      # hardcoded list of top-50 TWSE tickers (.TW suffix)
INDEX_PROXY       = "^TWII"           # TAIEX, for beta feature; 0050.TW as ETF proxy
LABEL_HORIZON     = 10               # trading days forward
REBALANCE_FREQ    = 5                # trade every 5 days
N_SPLITS          = 8                # walk-forward folds
EMBARGO_DAYS      = 5
N_QUANTILES       = 5                # quintile long-short (top/bottom 10 of 50 names)
COST_BPS          = 30               # round-trip cost; TW has ~0.3% sell tax + fees
SECTOR_NEUTRAL    = True
SEED              = 42
```

---

## 2. Stage 1 — Data ingestion (≈ half day)

**File:** `data/ingest.py`

**Tasks**

1. Define the universe as a _fixed hardcoded list_ of the **top 50 TWSE tickers** (paste from a current 0050.TW / TWSE market-cap ranking, append the `.TW` suffix). Keep the list in the repo so it's reproducible.
2. Pull adjusted daily OHLCV via `yfinance` (auto-adjusts splits/dividends). Also pull `^TWII` for the index/beta proxy.
3. Align all tickers to a common trading calendar; drop names with excessive missing history.
4. Cache to `data/cache/prices.parquet`. Treat raw pull as **immutable** — never overwrite.
5. Add a thin loader: if cache exists, read it; else pull and write.

**Function sketch**

```python
def load_prices(tickers, start, end, cache="data/cache/prices.parquet") -> pd.DataFrame:
    # returns tidy long frame: [date, ticker, open, high, low, close, volume]
```

**Gotcha to handle (and to say out loud in the interview)**

- **Survivorship bias.** Today's top-50 list excludes names that fell out of the ranking → inflated returns. Add a comment in code and a line in the README: _"Known limitation; production would use a point-in-time TWSE constituent history (e.g. TEJ)."_
- **Concentration.** TSMC (2330.TW) alone is a huge share of TWSE market cap; with only 50 names the cross-section is semiconductor-heavy. Flag this — it motivates the sector-neutralization step.
- Forward-fill prices **carefully** and **never** fill the forward-return target.

**Acceptance**

- [ ] Loading from cache is instant.
- [ ] No ticker has gaps that silently become fake returns.

---

## 3. Stage 2 — Feature / alpha engineering (≈ half day)

**File:** `features/alpha.py`

**Tasks**

1. Compute per-stock time-series features (all point-in-time, using only past data):
    - Momentum 12–1 month (skip the most recent month to avoid reversal contamination).
    - Short-term reversal (5-day return).
    - Realized volatility (21d, 63d).
    - Dollar-volume trend / liquidity.
    - Distance from moving averages (e.g. close / SMA50 − 1).
    - Optional: rolling beta to an index proxy (e.g. `^TWII` / `0050.TW`).
    - Target around 15–25 features. Resist adding more.
2. Build the **target**: forward return over `LABEL_HORIZON`, then convert to a **cross-sectional rank** within each date.
3. **Cross-sectional normalization:** each day, z-score or rank every feature across all names. This is the conceptual core — it turns absolute features into "how this stock looks vs its peers today."

**Function sketch**

```python
def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    # per-stock rolling features, then groupby('date') cross-sectional z-score/rank
    # returns [date, ticker, feat_1..feat_k, fwd_ret, fwd_rank, sector]

def add_forward_target(df, horizon):  # strictly future; no overlap with features
```

**Gotchas**

- The forward target leaks if you compute it before splitting time. Compute it, but make sure the splitter (Stage 3) purges around it.
- Drop the last `horizon` rows per ticker (no future to label them with).

**Acceptance**

- [ ] On any single date, every feature has mean ≈ 0 / is rank-uniform across names.
- [ ] No feature uses information from date > t at row t (eyeball a couple of names by hand).

---

## 4. Stage 3 — Predictive modeling + validation (≈ 1 day, the deepest stage)

**Files:** `model/validation.py`, `model/train.py`

This is the slide that separates you from the pack. Get it right.

### 4.1 Purged, embargoed walk-forward splitter

**File:** `model/validation.py`

Because a 10-day forward-return label means consecutive samples' label windows overlap, naive k-fold leaks the future into training. Purge training rows whose label window overlaps the test window; add an embargo gap.

```python
def purged_walk_forward(dates, n_splits, embargo_days, label_horizon):
    unique_days = np.sort(dates.unique())
    folds = np.array_split(unique_days, n_splits + 1)
    for k in range(1, len(folds)):
        test_days = folds[k]
        test_start, test_end = test_days[0], test_days[-1]
        purge_lo = test_start - pd.Timedelta(days=label_horizon + embargo_days)
        purge_hi = test_end   + pd.Timedelta(days=embargo_days)
        train_mask = (dates < purge_lo) | (dates > purge_hi)
        test_mask  = dates.isin(test_days)
        yield np.where(train_mask)[0], np.where(test_mask)[0]
```

> **Demo-the-failure trick:** also run a naive k-fold once and show that IC roughly doubles. Presenting the _inflated_ number as a cautionary tale is a power move.

### 4.2 Train / predict loop

**File:** `model/train.py`

**Tasks**

1. For each walk-forward fold: train LightGBM on the train slice, predict the test slice, collect out-of-sample predictions.
2. Target = `fwd_rank` (regression on the cross-sectional rank). Keep it simple; `lambdarank` is optional polish, not required.
3. Modest hyperparameters (shallow trees, strong regularization) — low signal-to-noise punishes over-capacity. Note this choice explicitly.
4. Concatenate all OOS predictions into one frame for the backtest.

**Function sketch**

```python
def run_walk_forward(features_df, feature_cols) -> pd.DataFrame:
    # returns [date, ticker, pred, fwd_ret, sector] — fully out-of-sample
```

**Acceptance**

- [ ] Predictions for a date are produced by a model that never saw that date or its label window.
- [ ] Feature importances are sane (momentum/vol dominate, not some leakage artifact).

---

## 5. Stage 4 — Backtest metrics (≈ half day)

**File:** `backtest/metrics.py`

Compute signal quality **before** PnL.

```python
def daily_ic(df, score="pred", fwd="fwd_ret"):
    return df.groupby("date").apply(
        lambda g: g[score].corr(g[fwd], method="spearman"))

def icir(ic_series):
    return ic_series.mean() / ic_series.std() * np.sqrt(252)
```

**Tasks**

- Report mean IC, ICIR, IC t-stat, and a rolling-IC plot.

**Acceptance**

- [ ] Mean IC and ICIR land in the sanity band (§0). If wildly high → leakage.

---

## 6. Stage 5–6 — Portfolio + execution/costs (≈ 1 day)

**Files:** `backtest/portfolio.py`, `backtest/engine.py`

### 6.1 Portfolio construction

- Each rebalance (`REBALANCE_FREQ`): rank by `pred`, long top quintile, short bottom quintile (~10 names per leg out of 50).
- Dollar-neutral, equal-weight (or rank-weight) within each leg.
- If `SECTOR_NEUTRAL`: demean predictions within sector first (use the TWSE/GICS sector tag), so the signal isn't a disguised semiconductor bet — important given TSMC + the chip cluster dominate this universe.
- Short-availability caveat: borrowing some TWSE names is hard/expensive in practice. Note that the long-short result is an idealization; a long-only top-quintile variant is the more realistic deployment.
- Cap per-name weight.

### 6.2 Vectorized backtester + costs

Write your own (~80 lines). Do **not** use a black-box library — owning the engine proves you understand it and avoids hidden lookahead.

- Hold weights between rebalances; compute daily portfolio return.
- **Transaction cost** = `COST_BPS` × turnover per rebalance. Turnover = sum of absolute weight changes.
- Produce **gross and net** equity curves.

**Function sketch**

```python
def build_weights(preds_df, n_quantiles, sector_neutral) -> pd.DataFrame:  # [date, ticker, weight]
def backtest(weights, returns, cost_bps) -> dict:
    # returns {gross_curve, net_curve, sharpe_gross, sharpe_net, mdd, turnover, calmar}
```

**Metrics to report:** annualized return, vol, gross Sharpe, **net Sharpe**, max drawdown, Calmar, average turnover, hit rate.

> **The single most important chart in the deck:** gross vs net equity curve. The drop is your honesty proof. If net is marginal, say the fix is _lower turnover_ (longer hold / signal smoothing), not a fancier model.

**Acceptance**

- [ ] Weights sum to ~0 each day (dollar-neutral).
- [ ] Net curve is strictly below gross; turnover is realistic (not 300%/day).

---

## 7. Stage 7 — Monitoring / drift (≈ half day, your home turf)

**File:** `monitoring/drift.py`

Mirror the three-ring detection framework from your MLOps prep — this closes the loop and ties in your last week of study.

**Tasks**

1. **Performance ring:** rolling 60-day IC — is the signal decaying over time? Plot it.
2. **Input-drift ring:** PSI or KS test per feature, early period vs late period — has the input distribution shifted (covariate shift = your Taiwan-roads problem restated)?
3. **Regime ring:** a simple flag (e.g. realized-vol percentile) so degraded stretches are explainable, not mysterious.
4. Generate an **Evidently** data-drift report → `reports/drift.html` as a real artifact to flash on a slide.

**Function sketch**

```python
def rolling_ic(ic_series, window=60): ...
def feature_psi(df_early, df_late, feature_cols): ...
def evidently_report(reference_df, current_df, path="reports/drift.html"): ...
```

**Defense line:** _"Same three layers I'd use in any production ML system — real-time proxy, windowed performance, batch statistical report. In markets the label is delayed, which is exactly why you can't lean on performance metrics alone and need the input-drift ring."_

---

## 8. Orchestration & reproducibility

**File:** `run_pipeline.ipynb`

This file show contain the following

1. load_prices
2. build_features
3. run_walk_forward
4. daily_ic/icir
5. build_weights/backtest
6. drift report

- Set all seeds (`SEED`).
- Use caching at every stage (data, features, predictions) so you can iterate on later stages without re-running everything.
- Use exsiting code from the modules, don't copy-paste logic into the notebook. The notebook is just the orchestration layer. If caching is done at module level, the notebook should run end-to-end in seconds on a cold cache, and near-instant on a warm cache.
- Use matplotlib or mplfinance for charts; make them clean and presentation-ready.
- Generate important chart into the /reports folder as HTML or PNG artifacts, so I can easily insert them into slides.

## 9. Suggested timeline (pre-interview)

| Day | Work                                | Output                          |
| --- | ----------------------------------- | ------------------------------- |
| 1   | Env + repo + ingestion + features   | Cached data, feature frame      |
| 2   | Validation splitter + LightGBM loop | OOS predictions, IC/ICIR        |
| 3   | Portfolio + cost-aware backtester   | Gross/net curves, metrics table |
| 4   | Drift monitoring + Evidently report | `drift.html`, rolling-IC plot   |
| 5   | Slides + rehearse §10 Q&A out loud  | 7-min demo, defensible answers  |

> If short on time, the irreducible core is: **purged walk-forward + IC + gross-vs-net costs**. Those three carry the credibility. Monitoring is the bonus that ties to your background.

---

## 10. Interview defense cheat-sheet (rehearse these aloud)

- **"Is this overfit?"** → "Likely degrades further OOS, which is why I lead with IC stability, not the PnL curve, and used purged walk-forward. Honest read: weak but real signal, most of it eaten by costs."
- **"What's your edge?"** → "I'm not claiming a deployable edge from a few days on free daily data. I'm showing I can build the loop that finds and validates edges without fooling myself."
- **"Why daily, not the HF data in the JD?"** → "HF data isn't free or laptop-scale, but the discipline — point-in-time features, purged validation, cost-aware backtest, drift monitoring — is identical. Horizon changes; rigor doesn't."
- **"Why LightGBM over a Transformer?"** → "Tabular cross-sectional features, few hundred names, low signal-to-noise — gradient boosting is the right tool and trains in seconds. Transformers earn their place with sequence/order-flow data, which I'd explore with your data."
- **"What with real infra?"** → "Point-in-time universe, intraday/order-flow features, a feature store + experiment tracking (MLflow/W&B), and this drift monitoring promoted to a live service."

---

## 11. Slide → artifact mapping

| Slide                 | Artifact from this repo                                       |
| --------------------- | ------------------------------------------------------------- |
| Data + bias controls  | universe list + README limitation note                        |
| Feature engineering   | feature list + cross-sectional code snippet                   |
| Modeling + validation | `purged_walk_forward` snippet + naive-vs-purged IC comparison |
| Results               | metrics summary table + rolling-IC plot                       |
| Costs                 | gross vs net equity curve                                     |
| Monitoring            | `reports/drift.html` screenshot                               |

---

## 12. Traps checklist (review before you call it done)

- [ ] No lookahead: features at t use only data ≤ t.
- [ ] Purge + embargo applied; naive-CV gap demonstrated.
- [ ] Forward target never forward-filled.
- [ ] Survivorship bias named, not hidden.
- [ ] Costs applied; gross vs net shown.
- [ ] Dollar-neutral weights; realistic turnover.
- [ ] Sector neutralization (so it's not a sector bet).
- [ ] Every number reproducible from `run_pipeline.py`.
- [ ] You can defend each choice in one sentence.
