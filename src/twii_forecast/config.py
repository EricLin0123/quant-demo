"""All tunable constants for the TWII next-day return forecasting pipeline.

Single source of truth: windows, split ratios, paths, RNG seed, thresholds.
Nothing downstream should hard-code a number that belongs here.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

RAW_PARQUET = DATA_RAW / "twii.parquet"
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
MODEL_PATH = DATA_PROCESSED / "lgbm_model.txt"
PARAMS_PATH = DATA_PROCESSED / "best_params.json"

for _d in (DATA_RAW, DATA_PROCESSED, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
TICKER = "^TWII"
START_DATE = "2005-01-01"          # modern regime; pre-2005 adds Asian-crisis noise
END_DATE = None                    # None => up to today

# Volume data-quality gate: usable if < 5% of rows are zero or NaN.
VOLUME_DEGENERATE_THRESHOLD = 0.05

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
LOOKBACK = 60                      # k: longest indicator window (warmup rows dropped)
MA_WINDOWS = (5, 10, 20, 60)       # moving-average / EMA windows
INTRADAY_MA_WINDOWS = (5, 10, 20)  # shorter windows for delta-based MAs
RSI_WINDOW = 14
WILLIAMS_WINDOW = 14
BB_WINDOW = 21
BB_STD = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MOMENTUM_WINDOW = 10
LAG_RETURNS = (0, 1, 2, 3, 5)      # r_t, r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}

# --------------------------------------------------------------------------- #
# Feature selection — temporal consistency analysis (AUC ~ 0.5 == stable)
# --------------------------------------------------------------------------- #
TCA_BLOCK_MONTHS = 3               # trimonthly blocks
TCA_TAU = 0.7                      # keep features with aggregated AUC <= tau
TCA_AGG = "mean"                   # "mean" or "max" aggregation across block pairs

# --------------------------------------------------------------------------- #
# Split (chronological, contiguous, never shuffled)
# --------------------------------------------------------------------------- #
TRAIN_FRAC = 0.85
VAL_FRAC = 0.05                    # => test is the most-recent 10%

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
SEED = 42
OBJECTIVE = "regression_l1"        # MAE — robust to leptokurtic return tails
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
N_RANDOM_SEARCH = 40               # tuning draws

# Random-search space (sampled with config.SEED for repeatability).
SEARCH_SPACE = {
    "num_leaves": (15, 255),
    "learning_rate": (0.005, 0.1),       # log-uniform
    "max_depth": (3, 12),
    "min_child_samples": (10, 120),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "lambda_l1": (1e-4, 10.0),           # log-uniform
    "lambda_l2": (1e-4, 10.0),           # log-uniform
}
