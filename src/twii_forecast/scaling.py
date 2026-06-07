"""RobustScaler (IQR) fit on TRAIN ONLY — the only leakage-free option.

    x' = (x - Q25_train) / (Q75_train - Q25_train)

Trees split on order not magnitude, so LightGBM does not strictly need this; we
apply it per spec so the feature matrix is model-agnostic if the model is ever
swapped. Features only — the target is never scaled.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import RobustScaler


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(X_train)
    return scaler


def transform(scaler: RobustScaler, X: pd.DataFrame) -> pd.DataFrame:
    arr = scaler.transform(X)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)
