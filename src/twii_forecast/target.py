"""The prediction target: next-day log return.

    r_{t+1} = ln( C_{t+1} / C_t )

Stationary, and left **unscaled** so MAE/RMSE come out in interpretable return
units. This is the only column in the pipeline that is intentionally future-facing;
everything in `features.py` is strictly causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def next_day_log_return(df: pd.DataFrame, close_col: str = "Close") -> pd.Series:
    """y_t = ln(C_{t+1} / C_t), aligned to time t (the day we predict *from*).

    The final row is NaN (no t+1 observed) and is dropped by `dataset.py`.
    """
    c = df[close_col]
    fwd = np.log(c.shift(-1) / c)
    fwd.name = "target_next_log_return"
    return fwd
