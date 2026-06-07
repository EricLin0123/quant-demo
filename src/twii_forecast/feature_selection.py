"""Feature selection by temporal consistency analysis (CalixBoost §3.2.3.2).

The paper weeds out columns that are *temporally inconsistent* — features whose
relationship with the target only holds inside the window you fit them on and
falls apart out-of-period. The probe is the ROC-AUC of a single feature treated
as a binary classifier of *up vs. down* days:

  * Split the series into consecutive 3-month ("trimonthly") periods. A monthly
    split is too small — ~20 business days per month is not enough points to
    assess a feature robustly — so three months is the smallest reliable block.
  * Learn the feature's predictive **direction** on a source period A (does a
    larger value mean a more likely up-day, or the reverse?).
  * Carry that direction to every other non-overlapping period B and measure
    ``roc_auc_score`` there. AUC ~ 0.5 is a coin flip; the paper's threshold is
    the minimal crossing at **AUC = 0.5**.
  * A feature whose mean out-of-period AUC stays at or above 0.5 generalised
    across time and is **kept**; one that sinks below 0.5 (it was right only
    where it was fit) is **dropped**.

Using the direction-from-A / score-on-B construction is exactly the AUC a
monotonic single-feature classifier (e.g. logistic regression) fit on A would
obtain on B, but with no fitting loop and no randomness — the score is computed
with scikit-learn's ``roc_auc_score`` as the paper specifies.

**Leakage note.** The paper runs this across all data. We restrict it to the
*training* split so the kept-feature decision never sees validation or test
rows — feature selection informed by the test regime would be leakage. The rest
of the pipeline then uses only the surviving columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config

logger = logging.getLogger(__name__)


def trimonthly_periods(index: pd.DatetimeIndex, months: int = config.TEMPORAL_PERIOD_MONTHS) -> np.ndarray:
    """Map each timestamp to a consecutive ``months``-long block id (0, 1, 2, ...).

    Blocks are anchored at the first calendar month in ``index`` so the first
    ``months`` calendar months are period 0, the next ``months`` are period 1,
    and so on — independent of how many trading days each block contains.
    """
    month_ordinal = index.year * 12 + (index.month - 1)
    return ((month_ordinal - month_ordinal.min()) // months).to_numpy()


def _binary_target(y: pd.Series) -> np.ndarray:
    """Up/down label for ROC-AUC: 1 if the next-day return is positive else 0."""
    return (y.to_numpy() > 0).astype(int)


def _cross_period_auc(x_a: np.ndarray, yb_a: np.ndarray,
                      x_b: np.ndarray, yb_b: np.ndarray) -> float | None:
    """AUC on period B using the feature direction learned on period A.

    Returns ``None`` when the AUC is undefined (a period with only one class, or
    a feature that is constant on the source period so no direction exists).
    """
    if len(np.unique(yb_a)) < 2 or len(np.unique(yb_b)) < 2:
        return None
    if np.all(x_a == x_a[0]):  # constant on A -> no learnable direction
        return None
    auc_a = roc_auc_score(yb_a, x_a)
    # If larger values mark *down* days on A, flip the sign before scoring B.
    score_b = x_b if auc_a >= 0.5 else -x_b
    return float(roc_auc_score(yb_b, score_b))


@dataclass
class TemporalConsistencyResult:
    """Per-feature trimonthly AUC summary and the kept/dropped partition."""

    summary: pd.DataFrame                 # index=feature; mean/min/std AUC, n_pairs
    kept: list[str]
    dropped: list[str]
    threshold: float
    n_periods: int
    period_pairs: int

    @property
    def auc_by_feature(self) -> pd.Series:
        return self.summary["mean_auc"]

    def report_frame(self) -> pd.DataFrame:
        """Tidy frame for display/CSV: AUC stats plus the keep/drop decision."""
        out = self.summary.copy()
        out["kept"] = out.index.isin(self.kept)
        return out.sort_values("mean_auc", ascending=False)


def analyze(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    months: int = config.TEMPORAL_PERIOD_MONTHS,
    threshold: float = config.TEMPORAL_AUC_THRESHOLD,
    min_period_rows: int = config.TEMPORAL_MIN_PERIOD_ROWS,
) -> TemporalConsistencyResult:
    """Run the trimonthly temporal-consistency analysis over every feature.

    ``X`` and ``y`` must share a ``DatetimeIndex`` (pass the *training* split).
    For each feature the AUC is averaged over all ordered non-overlapping period
    pairs (A -> B); a feature is kept iff that mean is ``>= threshold``.
    """
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("temporal consistency analysis requires a DatetimeIndex")

    yb = _binary_target(y)
    period_id = trimonthly_periods(X.index, months)

    # Keep only periods with enough rows to assess (a near-empty tail block at
    # the split boundary is not statistically meaningful).
    period_ids = [p for p in np.unique(period_id)
                  if (period_id == p).sum() >= min_period_rows]
    if len(period_ids) < 2:
        raise ValueError(
            f"need >= 2 trimonthly periods of >= {min_period_rows} rows; "
            f"got {len(period_ids)} from {len(X)} rows"
        )

    # Pre-slice per period once: feature matrix as ndarray + binary labels.
    masks = {p: (period_id == p) for p in period_ids}
    feat_cols = list(X.columns)
    Xv = X.to_numpy()
    col_idx = {c: i for i, c in enumerate(feat_cols)}
    per_period = {p: (Xv[masks[p]], yb[masks[p]]) for p in period_ids}

    pairs = list(permutations(period_ids, 2))  # ordered: train on A, score on B
    records: dict[str, list[float]] = {c: [] for c in feat_cols}
    for a, b in pairs:
        Xa, ya = per_period[a]
        Xb, yb_b = per_period[b]
        for c in feat_cols:
            j = col_idx[c]
            auc = _cross_period_auc(Xa[:, j], ya, Xb[:, j], yb_b)
            if auc is not None:
                records[c].append(auc)

    rows = {}
    for c in feat_cols:
        aucs = np.asarray(records[c], dtype=float)
        if aucs.size == 0:  # never assessable -> treat as failing (drop)
            rows[c] = {"mean_auc": np.nan, "min_auc": np.nan,
                       "std_auc": np.nan, "n_pairs": 0}
        else:
            rows[c] = {"mean_auc": float(aucs.mean()), "min_auc": float(aucs.min()),
                       "std_auc": float(aucs.std()), "n_pairs": int(aucs.size)}
    summary = pd.DataFrame.from_dict(rows, orient="index")

    keep_mask = summary["mean_auc"] >= threshold
    kept = summary.index[keep_mask.fillna(False)].tolist()
    dropped = [c for c in feat_cols if c not in kept]

    logger.info(
        "Temporal consistency: %d periods, %d ordered pairs, AUC>=%.2f -> "
        "kept %d / %d features (dropped: %s)",
        len(period_ids), len(pairs), threshold, len(kept), len(feat_cols),
        ", ".join(dropped) or "none",
    )
    return TemporalConsistencyResult(
        summary=summary, kept=kept, dropped=dropped, threshold=threshold,
        n_periods=len(period_ids), period_pairs=len(pairs),
    )


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs,
) -> tuple[list[str], TemporalConsistencyResult]:
    """Convenience wrapper: return ``(kept_columns, result)`` from the train split.

    If selection is disabled in config, all columns are kept and the analysis is
    skipped (returns an empty-summary result for transparency).
    """
    if not config.FEATURE_SELECTION_ENABLED:
        result = TemporalConsistencyResult(
            summary=pd.DataFrame(columns=["mean_auc", "min_auc", "std_auc", "n_pairs"]),
            kept=list(X_train.columns), dropped=[],
            threshold=config.TEMPORAL_AUC_THRESHOLD, n_periods=0, period_pairs=0,
        )
        return result.kept, result

    result = analyze(X_train, y_train, **kwargs)
    return result.kept, result
