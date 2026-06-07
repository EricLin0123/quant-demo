"""Temporal Consistency Analysis (CalixBoost §6 / Appendix B).

Drop features whose *distribution drifts across time* — a model trained on the past
cannot generalise on a non-stationary column. Drift is measured by how well a single
feature separates one time period from another, scored by ROC-AUC:

    AUC ~ 0.5  -> periods indistinguishable -> STABLE -> keep
    AUC -> 1   -> perfectly separable       -> DRIFT  -> drop

We use ``max(AUC, 1 - AUC)`` so the *direction* of separation doesn't matter (both
extremes are drift). The training history is cut into consecutive 3-month
(*trimonthly*) blocks; every non-overlapping pair of blocks is scored per feature,
the pairwise AUCs are aggregated, and features within ``tau`` of 0.5 survive.

Runs on the **training period only** — letting val/test influence selection would leak.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config

logger = logging.getLogger(__name__)


def make_trimonthly_blocks(
    index: pd.DatetimeIndex, block_months: int = config.TCA_BLOCK_MONTHS
) -> list[np.ndarray]:
    """Partition a DatetimeIndex into consecutive `block_months`-month blocks.

    Returns a list of positional-index arrays (one per non-empty block).
    """
    months = index.year * 12 + (index.month - 1)
    block_id = (months - months.min()) // block_months
    blocks: list[np.ndarray] = []
    for b in np.unique(block_id):
        pos = np.where(block_id == b)[0]
        if len(pos) > 0:
            blocks.append(pos)
    return blocks


def temporal_consistency(
    train_X: pd.DataFrame,
    feature_cols: list[str] | None = None,
    tau: float = config.TCA_TAU,
    agg: str = config.TCA_AGG,
    block_months: int = config.TCA_BLOCK_MONTHS,
) -> tuple[list[str], pd.DataFrame]:
    """Return (kept_features, auc_table).

    `auc_table` has one row per feature with its aggregated direction-agnostic AUC
    and the keep/drop decision, sorted most-stable first.
    """
    feature_cols = feature_cols or list(train_X.columns)
    blocks = make_trimonthly_blocks(train_X.index, block_months)
    if len(blocks) < 2:
        raise ValueError(f"need >=2 trimonthly blocks, got {len(blocks)}")
    logger.info(
        "temporal consistency: %d trimonthly blocks, %d block pairs, %d features",
        len(blocks), len(list(combinations(range(len(blocks)), 2))), len(feature_cols),
    )

    pairs = list(combinations(range(len(blocks)), 2))
    rows = []
    values = train_X[feature_cols].values
    col_idx = {c: i for i, c in enumerate(feature_cols)}

    for j in feature_cols:
        ci = col_idx[j]
        scores = []
        for a, b in pairs:
            xa = values[blocks[a], ci]
            xb = values[blocks[b], ci]
            x = np.concatenate([xa, xb])
            y = np.concatenate([np.zeros(len(xa)), np.ones(len(xb))])
            # A constant feature can't separate -> AUC undefined; treat as 0.5 (stable).
            if np.all(x == x[0]):
                scores.append(0.5)
                continue
            auc = roc_auc_score(y, x)
            scores.append(max(auc, 1.0 - auc))  # direction-agnostic
        agg_auc = float(np.mean(scores)) if agg == "mean" else float(np.max(scores))
        rows.append((j, agg_auc))

    auc_table = pd.DataFrame(rows, columns=["feature", "agg_auc"])
    auc_table["keep"] = auc_table["agg_auc"] <= tau
    auc_table = auc_table.sort_values("agg_auc").reset_index(drop=True)

    kept = auc_table.loc[auc_table["keep"], "feature"].tolist()
    dropped = auc_table.loc[~auc_table["keep"], "feature"].tolist()
    logger.info("temporal consistency: kept %d / %d (tau=%.2f, agg=%s)",
                len(kept), len(feature_cols), tau, agg)
    if dropped:
        logger.info("dropped (drift, AUC>%.2f): %s", tau, dropped)

    out = config.REPORTS / "temporal_consistency.csv"
    auc_table.to_csv(out, index=False)
    logger.info("wrote %s", out)
    return kept, auc_table
