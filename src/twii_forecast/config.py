"""All project constants in one place: windows, paths, split ratios, seed.

Nothing here imports heavy libraries — keep it cheap to import everywhere.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TICKER = "^TWII"          # Taiwan Weighted Index (TAIEX) on Yahoo Finance
BENCHMARK_START = "2015-01-01"  # ~2015 -> present (modern regime)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

RAW_PARQUET = RAW_DIR / "twii.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
MODEL_PATH = MODELS_DIR / "lgbm.txt"
PARAMS_PATH = MODELS_DIR / "best_params.json"

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering windows
# ---------------------------------------------------------------------------
MA_SHORT_WINDOWS = (5, 10, 20)        # intraday / delta moving averages
MA_PRICE_WINDOWS = (5, 10, 20, 60)    # MA / EMA on close
RSI_WINDOW = 14
WILLIAMS_WINDOW = 14
BB_WINDOW = 21
MACD_FAST = 12
MACD_SLOW = 26
MOMENTUM_LAG = 10
LAG_RETURNS = (0, 1, 2, 3, 5)         # r_t, r_{t-1}, ... (0 == current)
WARMUP = 60                            # longest window; drop these warmup rows

# ---------------------------------------------------------------------------
# Volume data-quality gate
# ---------------------------------------------------------------------------
VOLUME_DEGENERATE_THRESHOLD = 0.05    # usable if (frac_zero + frac_nan) < 5%

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.85
VAL_FRAC = 0.05                        # -> val ends at 0.90
# test is the remaining 0.10

# ---------------------------------------------------------------------------
# Model / tuning
# ---------------------------------------------------------------------------
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
N_SEARCH_DRAWS = 40
OBJECTIVE = "regression_l1"           # MAE; robust to fat-tailed returns

TARGET_COL = "target_logret_next"
