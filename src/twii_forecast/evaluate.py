"""Metrics + baselines — the actual deliverable.

Every model metric is reported next to two naive baselines so the numbers mean
something:

  1. Persistence  : r_hat = 0           (MAE = mean|r|, the number to beat)
  2. Historical   : r_hat = mean(r_train) (a constant)

Reality check: out-of-sample R^2 on daily returns is expected near zero or
negative, and 51-54% directional accuracy is a *good* result. A dramatically
higher number on one split is leakage, not alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    name: str
    mae: float
    rmse: float
    dir_acc: float
    ic: float
    r2: float

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "DirAcc": self.dir_acc,
            "IC": self.ic,
            "R2": self.r2,
        }


def _metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    # Directional accuracy. Treat exact-zero predictions as "no call" -> exclude
    # so a constant 0-baseline isn't credited with a coin-flip it never made.
    mask = y_pred != 0
    if mask.any():
        dir_acc = float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))
    else:
        dir_acc = float("nan")

    # IC = correlation(pred, actual). Undefined if pred is constant.
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        ic = float("nan")
    else:
        ic = float(np.corrcoef(y_pred, y_true)[0, 1])

    # Out-of-sample R^2 vs. the realised variance of the target.
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return Metrics(name, mae, rmse, dir_acc, ic, r2)


def evaluate(y_test: pd.Series, y_pred: np.ndarray,
             y_train: pd.Series) -> pd.DataFrame:
    """Return a tidy table: model vs. both baselines on every metric."""
    yt = y_test.to_numpy()

    rows = [
        _metrics("LightGBM", yt, np.asarray(y_pred)),
        _metrics("Persistence (0)", yt, np.zeros_like(yt)),
        _metrics("Historical mean", yt, np.full_like(yt, float(y_train.mean()))),
    ]
    return pd.DataFrame([r.as_row() for r in rows]).set_index("model")
