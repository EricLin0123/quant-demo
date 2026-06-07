# TWII Next-Day Return Forecasting

A single **LightGBM** regressor that predicts the **next-day log return** of the
Taiwan Weighted Index (TAIEX, `^TWII`) from causal technical indicators. The
deliverable is *predictive power measured against naive baselines* — not a trading
book.

> **Honest scope.** Out-of-sample R² on daily returns is expected near zero or
> negative, and 51–54% directional accuracy is a *good* result. A dramatically
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
  scaling.py   RobustScaler (IQR), fit on TRAIN only
  model.py     LightGBM + early stopping + random search
  evaluate.py  MAE / RMSE / DirAcc / IC / R² vs. baselines
  monitor.py   evidently train-vs-test drift report
  plots.py     candles, pred-vs-actual, residuals, next-day directional accuracy
notebooks/run_pipeline.ipynb   thin end-to-end orchestration
tests/test_leakage.py          correctness gate (no future bleed)
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
   series at *t* does not change the feature row at *t*).
2. **Every metric is reported next to its baseline** — *persistence* (`r̂ = 0`) and
   *historical mean* (`r̂ = mean(r_train)`).

## Notes on the data

- `^TWII` has **no real Adjusted Close** (an index has no dividend reinvestment), so
  `AdjClose := Close` and the duplicate AC-based indicators are dropped.
- Yahoo's index volume can be degenerate; `data.validate_volume` gates it at runtime
  (<5% zero/NaN to keep volume features). With the current pull it passes.
