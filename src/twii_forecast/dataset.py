"""Assemble the aligned (X, y) modelling table.

Builds causal features (`features.py`) and the future target (`target.py`), joins
them on Date, drops the ~60 warmup rows (NaNs from the longest indicator window) and
the final row (no t+1 target). The result is a single tidy DataFrame whose feature
columns are all defined and whose target is the next-day log return.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config, data, features, target

logger = logging.getLogger(__name__)

TARGET_COL = "target_next_log_return"


def build_dataset(
    df_ohlcv: pd.DataFrame | None = None,
    volume_ok: bool | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (table, feature_cols).

    `table` has the feature columns plus `TARGET_COL`, indexed by Date, with no NaNs.
    """
    if df_ohlcv is None or volume_ok is None:
        df_ohlcv, volume_ok = data.load(use_cache=use_cache)

    X = features.build_features(df_ohlcv, volume_ok=volume_ok)
    y = target.next_day_log_return(df_ohlcv)
    feature_cols = features.feature_columns(X)

    table = X.copy()
    table[TARGET_COL] = y

    n_before = len(table)
    table = table.dropna(axis=0, how="any")
    n_after = len(table)
    logger.info(
        "dataset: %d rows -> %d after dropping warmup/tail NaNs (%d features)",
        n_before, n_after, len(feature_cols),
    )

    table.to_parquet(config.FEATURES_PARQUET)
    return table, feature_cols
