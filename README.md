# TWII Next-Day Return Forecasting

A single **LightGBM** regressor that predicts the **next-day log return** of the
Taiwan Weighted Index (TAIEX, `^TWII`) from causal technical indicators. The
deliverable is _predictive power measured against naive baselines_ — not a trading
book.

> **Honest scope.** Out-of-sample R² on daily returns is expected near zero or
> negative, and 51–54% directional accuracy is a _good_ result. A dramatically
> higher number on one split is leakage, not alpha. Every metric is reported next
> to a baseline so the numbers mean something.

## Layout

```
src/twii_forecast/
  config.py    constants: windows, split ratios, paths, seed
  data.py      yfinance pull + volume data-quality gate
  features.py  causal/trailing technical indicators
  target.py    next-day log return  r_{t+1} = ln(C_{t+1}/C_t)
  dataset.py   assemble X/y, drop warmup + tail NaNs
  split.py     chronological 85/5/10
  feature_selection.py  temporal consistency analysis (trimonthly AUC)
  scaling.py   RobustScaler (IQR), fit on TRAIN only
  model.py     LightGBM + early stopping + random search
  evaluate.py  MAE / RMSE / DirAcc / IC / R² vs. baselines
  monitor.py   evidently train-vs-test drift report
  plots.py     candles, pred-vs-actual, residuals, next-day directional accuracy
notebooks/run_pipeline.ipynb   thin end-to-end orchestration
tests/test_leakage.py            correctness gate (no future bleed)
tests/test_feature_selection.py  temporal consistency analysis behaviour
```

## Run

```bash
uv sync
uv run pytest                        # leakage gate — must pass
uv run jupyter lab notebooks/run_pipeline.ipynb
```

## The two non-negotiable correctness gates

1. **`tests/test_leakage.py` passes** — every rolling op is trailing; the target is
   strictly future (`ln(C_{t+1}/C_t)`); features are provably causal (truncating the
   series at _t_ does not change the feature row at _t_).
2. **Every metric is reported next to its baseline** — _persistence_ (`r̂ = 0`) and
   _historical mean_ (`r̂ = mean(r_train)`).

## Feature selection: temporal consistency analysis (CalixBoost §3.2.3.2)

Before training, `feature_selection.py` weeds out **temporally inconsistent** columns —
features whose *distribution drifts across time*, because a model trained on the past
won't generalise on a non-stationary column. The training series is split into
consecutive 3-month (_trimonthly_) periods; ~20 business days a month is too few for a
monthly split to be significant. For every **non-overlapping pair** of periods, each
feature is scored by how well it alone separates one period from the other —
`sklearn.metrics.roc_auc_score`, taken as `max(AUC, 1 − AUC)` so the direction of
separation doesn't matter. Reading is inverted from the usual convention: **AUC ≈ 0.5
is the *good* outcome** — the periods are indistinguishable, so the feature is stable and
is **kept**; an AUC driven toward 1 means the feature alone tells the periods apart
(drift), so it is **dropped**. The pairwise AUCs are averaged per feature and thresholded
at **τ = 0.7** (the paper treats 0.7–0.9 as "strong separation"). On the current pull
this keeps 23 of 36 features and drops the non-stationary price-level block (the MAs,
EMAs, Bollinger bands and MACD that trend with the index). The analysis runs on **train
only** (selecting on val/test would leak), the surviving subset flows into scaling,
training and evaluation, and the full per-feature AUC table is written to
`reports/temporal_consistency.csv`. A whole-distribution `evidently` train-vs-test drift
report (`monitor.py`) is written alongside it as `reports/drift_report.html`.

## Notes on the data

- `^TWII` has **no real Adjusted Close** (an index has no dividend reinvestment), so
  `AdjClose := Close` and the duplicate AC-based indicators are dropped.
- Yahoo's index volume can be degenerate; `data.validate_volume` gates it at runtime
  (<5% zero/NaN to keep volume features). With the current pull it passes.

## Reference

- [CalixBoost](https://aircconline.com/csit/papers/vol12/csit121009.pdf)
- [Shapley Additive Explanations (SHAP)](https://www.youtube.com/watch?v=VB9uV-x0gtg)
- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/pdf/1705.07874)
