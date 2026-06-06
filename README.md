# quant-demo

A self-contained, laptop-scale ML quant research loop on the **TWSE top-50**
universe — built to demonstrate *methodological rigor*, not to find deployable
alpha. It stands up the full pipeline (data → alpha → model → backtest →
portfolio → cost-aware execution → drift monitoring) and stress-tests it
honestly: purged walk-forward validation, cross-sectional normalization, real
transaction costs, and a three-ring drift monitor. The guiding principle is that
a modest, honest result with airtight methodology beats a Sharpe-of-3 fantasy.

## Run it

```bash
uv run run_pipeline.py            # reproduces every number + chart (warm cache ≈ 5s)
uv run run_pipeline.py --cold     # ignore caches, rebuild from a fresh data pull
uv run run_pipeline.py --no-html  # skip the (slower) Evidently HTML report
```

Every stage is build-or-load and immutably cached, and all randomness is seeded
(`config.SEED`), so re-runs are instant and every artifact traces back to this
one entry point. Each module also runs standalone (e.g. `uv run
backtest/engine.py`) printing its own acceptance checks.

## Pipeline

| Stage | Module | Output |
| --- | --- | --- |
| 1 Data | `data/ingest.py` | adjusted OHLCV parquet, common calendar |
| 2 Features | `features/alpha.py` | ~18 point-in-time features, cross-sectionally z-scored each day |
| 3 Model | `model/validation.py`, `model/train.py` | **purged + embargoed** walk-forward LightGBM, OOS predictions |
| 4 Metrics | `backtest/metrics.py` | mean IC, ICIR, t-stat, rolling-IC plot |
| 5–6 Backtest | `backtest/portfolio.py`, `backtest/engine.py` | dollar-neutral quintile long-short, own vectorized engine, **gross vs net** curves |
| 7 Monitoring | `monitoring/drift.py` | three-ring drift: rolling IC, PSI/KS, regime flag, Evidently `reports/drift.html` |

Production logic lives in modules; `notebooks/` is exploration only.

## Headline results (out-of-sample)

- **Mean daily IC ≈ 0.035**, **ICIR ≈ 0.9** (overlap-adjusted). The naive
  `√252` ICIR reads ≈ 2.9, but daily ICs share 9/10 of their 10-day label window
  (lag-1 autocorr ≈ 0.8), so we annualize over *effective* independent periods —
  the honest number is ~0.9, squarely in the sanity band.
- **Long-only top-quintile**: gross Sharpe ≈ 1.2, **net ≈ 0.7** — the realistic,
  shortable deployment.
- **Long-short** (idealized): the gross signal is positive but thin, and the
  **net Sharpe is eaten by ~10%/yr turnover cost**. The fix is a longer hold /
  signal smoothing, *not* a fancier model — that gross-vs-net gap is the point.

## Known limitations (named, not hidden)

- **Survivorship bias**: the universe is *today's* top-50 ranking, so names that
  dropped out are excluded — this inflates historical returns. Production would
  use a point-in-time TWSE constituent history (e.g. TEJ).
- **Concentration**: TSMC + the semiconductor cluster dominate TWSE market cap,
  which is exactly why predictions are sector-neutralized before ranking.
- **Short availability**: borrowing some TWSE names is hard/expensive, so the
  long-short book is an idealization; long-only is the deployable variant.
- **Scope**: daily data only. The discipline (point-in-time features, purged
  validation, cost-aware backtest, drift monitoring) is identical at higher
  frequency; the horizon changes, the rigor doesn't.
