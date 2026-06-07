"""Chronological, contiguous split: oldest 85% train / 5% val / newest 10% test.

No shuffling, ever — this is time series. The most recent window is the test set
so the evaluation regime is modern.
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


def chronological_split(frame: pd.DataFrame) -> Split:
    n = len(frame)
    i_tr = int(config.TRAIN_FRAC * n)
    i_va = int((config.TRAIN_FRAC + config.VAL_FRAC) * n)
    train, val, test = frame.iloc[:i_tr], frame.iloc[i_tr:i_va], frame.iloc[i_va:]
    logger.info(
        "Split: train=%d (%s..%s) val=%d test=%d (%s..%s)",
        len(train), train.index[0].date(), train.index[-1].date(),
        len(val), len(test), test.index[0].date(), test.index[-1].date(),
    )
    return Split(train=train, val=val, test=test)
