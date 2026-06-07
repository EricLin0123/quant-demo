"""Assemble (X, y): join features and target, drop warmup + tail NaNs.

The last row has no next-day return (target is NaN) and the first ~60 rows are
warmup NaNs from the longest rolling window. Both are dropped here so every row
handed to the model is complete.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config, features, target

logger = logging.getLogger(__name__)


def build_dataset(df: pd.DataFrame, use_volume: bool = True) -> pd.DataFrame:
    """Return a single frame: feature columns + the target column, NaN-free."""
    X = features.build_features(df, use_volume=use_volume)
    y = target.next_day_log_return(df)

    full = X.join(y)

    # Drop the explicit warmup window first (longest indicator = 60d), then any
    # residual NaNs (incl. the final row whose target is undefined).
    before = len(full)
    full = full.iloc[config.WARMUP:]
    full = full.dropna()
    logger.info(
        "Dataset: %d rows after warmup+dropna (from %d), %d feature cols",
        len(full), before, full.shape[1] - 1,
    )
    return full


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from the target column."""
    y = frame[config.TARGET_COL]
    X = frame.drop(columns=[config.TARGET_COL])
    return X, y
