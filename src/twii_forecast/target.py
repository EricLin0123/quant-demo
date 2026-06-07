"""Prediction target: next-day log return.

    r_{t+1} = ln(C_{t+1} / C_t)

Aligned to row ``t`` so that ``X`` at ``t`` (features computed from info <= t)
predicts the return realised *over the next day*. Left unscaled so MAE/RMSE come
out in interpretable return units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def next_day_log_return(df: pd.DataFrame) -> pd.Series:
    c = df["Close"]
    logret = np.log(c / c.shift(1))   # return realised on day t (uses C_{t-1}, C_t)
    target = logret.shift(-1)         # bring day t+1's return back onto row t
    target.name = config.TARGET_COL
    return target
