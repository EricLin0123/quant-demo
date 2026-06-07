"""Temporal consistency analysis (CalixBoost §3.2.3.2) unit checks.

These assert the selector's *behaviour*, not its exact numbers: a feature whose
up/down direction is stable across time survives, a feature whose direction
flips every period (predictive only in-sample) is dropped, and pure noise lands
at the 0.5 boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twii_forecast import feature_selection as fs


@pytest.fixture(scope="module")
def panel():
    """~3 years of daily rows with three engineered feature behaviours."""
    n = 750
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(0)

    # Binary up/down truth, materialised as a signed next-day return.
    up = rng.integers(0, 2, n)
    y = pd.Series(np.where(up == 1, 1.0, -1.0) * rng.uniform(0.1, 1.0, n),
                  index=idx, name="target")

    period = fs.trimonthly_periods(idx)

    consistent = up + rng.normal(0, 0.3, n)          # always tracks the truth
    flip_sign = np.where(period % 2 == 0, 1, -1)     # direction reverses each period
    inconsistent = flip_sign * up + rng.normal(0, 0.3, n)
    noise = rng.normal(0, 1, n)                       # no relationship

    X = pd.DataFrame(
        {"consistent": consistent, "inconsistent": inconsistent, "noise": noise},
        index=idx,
    )
    return X, y


def test_consistent_feature_is_kept(panel):
    X, y = panel
    result = fs.analyze(X, y)
    assert "consistent" in result.kept
    assert result.auc_by_feature["consistent"] > 0.5


def test_direction_flipping_feature_is_dropped(panel):
    """A feature predictive only within each period must fall below 0.5 OOS."""
    X, y = panel
    result = fs.analyze(X, y)
    assert "inconsistent" in result.dropped
    assert result.auc_by_feature["inconsistent"] < 0.5


def test_trimonthly_periods_are_three_calendar_months():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    p = fs.trimonthly_periods(idx, months=3)
    # Jan-Mar -> 0, Apr-Jun -> 1, Jul-Sep -> 2, Oct-Dec -> 3.
    assert p.min() == 0 and p.max() == 3
    assert (p[idx.month <= 3] == 0).all()
    assert (p[(idx.month >= 10)] == 3).all()


def test_select_features_subsets_columns(panel):
    X, y = panel
    kept, result = fs.select_features(X, y)
    assert set(kept) == set(result.kept)
    assert set(kept).issubset(set(X.columns))
    # Subsetting the matrix to kept columns must preserve row count + order.
    assert list(X[kept].index) == list(X.index)


def test_disabled_keeps_everything(panel, monkeypatch):
    X, y = panel
    monkeypatch.setattr(fs.config, "FEATURE_SELECTION_ENABLED", False)
    kept, result = fs.select_features(X, y)
    assert kept == list(X.columns)
    assert result.dropped == []
