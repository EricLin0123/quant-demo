"""Per-next-day metrics + naive baselines — the actual deliverable.

Every metric is computed directly on the set of one-step-ahead predictions
``{(r_hat_{t+1}, r_{t+1})}`` over the test window. We never chain predictions into a
multi-step forecast and never compound r_hat into a reconstructed price level for
scoring — that would accumulate error and hide per-day accuracy. Directional accuracy
is the headline number.

Every model metric is printed next to two baselines:
  1. persistence:      r_hat = 0
  2. historical mean:   r_hat = mean(r_train)   (constant)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    mae: float
    rmse: float
    dir_acc: float
    ic: float
    r2: float

    def as_row(self) -> dict:
        return asdict(self)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R^2 vs the mean of y_true. Expect ~0 or negative on returns."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of days the predicted sign matches the realised sign.

    Zero-return days (sign 0) are excluded from the denominator so a degenerate
    all-zero predictor doesn't get spuriously credited on flat days.
    """
    mask = y_true != 0.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    dir_acc = _directional_accuracy(y_true, y_pred)
    # IC is undefined if a predictor is constant (zero variance).
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        ic = float("nan")
    else:
        ic = float(np.corrcoef(y_pred, y_true)[0, 1])
    return Metrics(mae, rmse, dir_acc, ic, _r2(y_true, y_pred))


def evaluate(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    """Return a tidy comparison table: model vs. persistence vs. historical-mean."""
    y_test = np.asarray(y_test, dtype=float)
    n = len(y_test)

    persistence = np.zeros(n)
    hist_mean = np.full(n, float(np.mean(y_train)))

    table = pd.DataFrame(
        {
            "model": compute_metrics(y_test, y_pred).as_row(),
            "persistence (r=0)": compute_metrics(y_test, persistence).as_row(),
            "historical_mean": compute_metrics(y_test, hist_mean).as_row(),
        }
    ).T
    table = table[["mae", "rmse", "dir_acc", "ic", "r2"]]
    return table


def direction_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exclude_zero: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Binary up/down confusion matrix for the directional call.

    Positive class is "up" (return > 0). Days with an exactly-flat actual return are
    excluded by default (consistent with `dir_acc`, which has no defined direction to
    score on those). Returns (confusion_df, metrics) where confusion_df is laid out
    actual (rows) x predicted (cols), both ordered [up, down].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if exclude_zero:
        mask = y_true != 0.0
        y_true, y_pred = y_true[mask], y_pred[mask]

    actual_up = y_true > 0.0
    pred_up = y_pred > 0.0

    tp = int(np.sum(actual_up & pred_up))     # actual up,   predicted up
    fn = int(np.sum(actual_up & ~pred_up))    # actual up,   predicted down
    fp = int(np.sum(~actual_up & pred_up))    # actual down, predicted up
    tn = int(np.sum(~actual_up & ~pred_up))   # actual down, predicted down

    cm = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["actual_up", "actual_down"],
        columns=["pred_up", "pred_down"],
    )

    total = tp + tn + fp + fn

    def safe_div(a: int, b: int) -> float:
        return a / b if b else float("nan")

    metrics = {
        "n": total,
        "accuracy": safe_div(tp + tn, total),
        "precision_up": safe_div(tp, tp + fp),
        "recall_up": safe_div(tp, tp + fn),
        "precision_down": safe_div(tn, tn + fn),
        "recall_down": safe_div(tn, tn + fp),    # specificity
        "pred_up_rate": safe_div(tp + fp, total),
        "actual_up_rate": safe_div(tp + fn, total),
    }
    f1 = safe_div(2 * metrics["precision_up"] * metrics["recall_up"],
                  metrics["precision_up"] + metrics["recall_up"])
    metrics["f1_up"] = f1
    return cm, metrics


def format_table(table: pd.DataFrame) -> str:
    fmt = table.copy()
    for c in ("mae", "rmse"):
        fmt[c] = fmt[c].map(lambda v: f"{v:.6e}")
    for c in ("dir_acc", "ic", "r2"):
        fmt[c] = fmt[c].map(lambda v: f"{v:+.4f}")
    return fmt.to_string()
