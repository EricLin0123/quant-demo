"""RobustScaler (IQR), fit on TRAIN only — the single leakage-free choice.

    x' = (x - Q25_train) / (Q75_train - Q25_train)

LightGBM splits on order, not magnitude, so trees don't strictly need this. We apply
it anyway per spec and to keep the feature matrix model-agnostic if the model is ever
swapped. The **target is never scaled** (MAE/RMSE stay in return units).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import RobustScaler


def fit_scaler(train_X: pd.DataFrame) -> RobustScaler:
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(train_X.values)
    return scaler


def apply_scaler(scaler: RobustScaler, X: pd.DataFrame) -> pd.DataFrame:
    arr = scaler.transform(X.values)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)
