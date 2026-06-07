"""Behaviour checks for temporal consistency analysis.

A stationary (regime-invariant) feature should survive; a feature with an obvious
secular trend should be flagged as drifting and dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twii_forecast import feature_selection as fs


def _daily_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=n)


def test_trimonthly_blocks_partition_fully():
    idx = _daily_index(400)
    blocks = fs.make_trimonthly_blocks(idx, block_months=3)
    assert len(blocks) >= 2
    # blocks partition all positions exactly once
    covered = np.sort(np.concatenate(blocks))
    np.testing.assert_array_equal(covered, np.arange(len(idx)))


def test_stationary_kept_trend_dropped():
    rng = np.random.default_rng(0)
    n = 600
    idx = _daily_index(n)
    df = pd.DataFrame(
        {
            "stationary": rng.normal(0.0, 1.0, n),   # no drift -> AUC ~ 0.5 -> keep
            "trend": np.linspace(0.0, 50.0, n),      # strong drift -> AUC ~ 1 -> drop
        },
        index=idx,
    )
    kept, table = fs.temporal_consistency(df, tau=0.7, agg="mean")
    auc = table.set_index("feature")["agg_auc"]
    assert "stationary" in kept
    assert "trend" not in kept
    assert auc["trend"] > auc["stationary"]
    assert auc["stationary"] < 0.7
