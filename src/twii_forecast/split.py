"""Chronological, contiguous split — oldest 85% train, next 5% val, last 10% test.

No shuffling, ever: the test window is the most-recent slice of history so the
evaluation mimics genuine out-of-sample forecasting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def span(d: pd.DataFrame) -> str:
            return f"{d.index.min().date()}..{d.index.max().date()} (n={len(d)})"
        return (
            f"train {span(self.train)} | val {span(self.val)} | test {span(self.test)}"
        )


def chronological_split(
    table: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> Split:
    """Split a Date-sorted table into contiguous train/val/test blocks."""
    table = table.sort_index()
    n = len(table)
    i_tr = int(train_frac * n)
    i_va = int((train_frac + val_frac) * n)
    sp = Split(table.iloc[:i_tr], table.iloc[i_tr:i_va], table.iloc[i_va:])
    logger.info("split: %s", sp.describe())
    return sp
