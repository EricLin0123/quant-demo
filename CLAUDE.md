# CLAUDE.md — guide for coding agents

TWII next-day return forecasting: a single **LightGBM** regressor predicting the
next-day log return of `^TWII` from causal technical indicators. The deliverable is
**honest predictive power measured against naive baselines**, not a trading book.

## Read first
- **`plan.md`** — authoritative build spec (scope, methods, formulas). The source of truth.
- **`README.md`** — repo layout, run instructions, feature-selection rationale.
- **This file** — run gotchas + current status. Keep the *Status* section updated.

## Run
```bash
uv sync
uv run pytest                       # correctness gates — must pass (7 tests)
PYTHONPATH=src uv run python -c "from twii_forecast import pipeline; r = pipeline.run()"
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/run_pipeline.ipynb
```
- **Imports need `src` on the path.** Use `PYTHONPATH=src` for scripts; the notebook's
  first cell bootstraps it; pytest reads `pythonpath=["src"]` from `pyproject.toml`.
- **After editing `src/`, restart any running Jupyter kernel.** Stale modules cached in
  `sys.modules` cause confusing `AttributeError`s (e.g. "config has no attribute X").
- Raw data is cached at `data/raw/twii.parquet`; `reports/` holds generated artifacts.
  Both `data/` and `reports/` are gitignored.

## Architecture
All logic lives in `src/twii_forecast/`; `pipeline.run()` orchestrates; the notebook is
thin (calls `pipeline.run()` and renders). Flow:
`config → data (+volume gate) → features (causal) → target → dataset → split →
feature_selection (train-only temporal-consistency AUC) → scaling (fit-on-train) →
model (LGBM + random search + early stopping) → evaluate (vs baselines) →
explain (SHAP) → monitor (drift) → plots`.

## Non-negotiable correctness gates
- `tests/test_leakage.py` passes — truncation-invariance (no future bleed); target is
  strictly future `ln(C_{t+1}/C_t)`.
- Feature selection runs on **train only**.
- Every metric is reported next to its baseline; scoring is **per next-day prediction,
  never a compounded price level**.

## Current status — UPDATE THIS as work progresses
- Pipeline complete; tests green. Latest full run (test n=519, 2024-04→2026-06):
  MAE ~1.09e-2 (edges persistence), DirAcc ~57% (≈ the always-up baseline's 56.8%),
  IC ~+0.05, R² ≈ 0. Honest, low-signal result exactly as `plan.md` predicted.
- **Open thread — model shrinks to a near-constant predictor.** Predicted returns span
  ±~8e-4 vs actuals ±2.5e-2; it predicts "up" ~91% of days. Diagnosed as the *correct*
  response of L1/MAE + early-stopping + regularization to a near-noise target (inflating
  predictions to actual scale strictly worsens MAE/RMSE) — **not a bug**. A directional
  confusion matrix + per-day actual-vs-predicted plot were added to the notebook to make
  this visible.
  **Hold: "report, don't fix" — awaiting user decision** on whether to trade accuracy for
  more expressive predictions (switch to L2/`huber`, relax early stopping/regularization).
