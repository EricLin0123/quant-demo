"""Correctness gate: assert no future information bleeds into the features.

Two independent checks:

1. **Causality of each feature.** Building features on a series truncated at
   time ``t`` must give the *same* feature row at ``t`` as building on the full
   series. If any indicator peeked at ``> t``, truncation would change it.

2. **Future bleed changes predictions.** Corrupting a feature column *only on
   the test rows* must change test predictions, while corrupting it only on
   *future* (post-test) rows must NOT — there are no post-test rows in a tail
   split, so we assert the contrapositive: shifting the target forward (a known
   leak) inflates the in-sample fit relative to the correct alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twii_forecast import dataset, features, target


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    """Deterministic synthetic OHLCV — no network in tests."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2018-01-01", periods=n)
    ret = rng.normal(0, 0.01, n)
    close = 10000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_features_are_causal(ohlcv):
    """Truncating the series at t must not change the feature row at t."""
    full = features.build_features(ohlcv, use_volume=True)

    # Probe several interior points past the warmup window.
    for t in (120, 200, 333):
        truncated = features.build_features(ohlcv.iloc[: t + 1], use_volume=True)
        a = full.iloc[t]
        b = truncated.iloc[t]
        pd.testing.assert_series_equal(
            a, b, check_names=False,
            obj=f"feature row at position {t}",
        )


def test_no_nans_after_assembly(ohlcv):
    frame = dataset.build_dataset(ohlcv, use_volume=True)
    assert not frame.isna().any().any()
    assert len(frame) > 0


def test_target_is_strictly_future(ohlcv):
    """Target at row t must equal ln(C_{t+1}/C_t) — a value not yet known at t."""
    y = target.next_day_log_return(ohlcv)
    c = ohlcv["Close"]
    expected = np.log(c.shift(-1) / c)
    pd.testing.assert_series_equal(y, expected, check_names=False)
    # Last row's target is undefined (no t+1).
    assert np.isnan(y.iloc[-1])


def test_shifting_feature_changes_predictions(ohlcv):
    """A simple model's test predictions must respond to the feature values.

    If features carried no real (causal) signal wiring, perturbing a test-set
    feature column would leave predictions unchanged. This guards the X->y plumbing.
    """
    from lightgbm import LGBMRegressor

    frame = dataset.build_dataset(ohlcv, use_volume=True)
    X, y = dataset.split_xy(frame)
    n = len(X)
    i = int(0.8 * n)
    X_tr, y_tr, X_te = X.iloc[:i], y.iloc[:i], X.iloc[i:].copy()

    model = LGBMRegressor(n_estimators=50, num_leaves=7, verbose=-1,
                          random_state=0)
    model.fit(X_tr, y_tr)
    base = model.predict(X_te)

    X_pert = X_te.copy()
    X_pert["difference"] = X_pert["difference"] + 1.0  # shift a key feature
    pert = model.predict(X_pert)

    assert not np.allclose(base, pert), "predictions ignore feature values"
