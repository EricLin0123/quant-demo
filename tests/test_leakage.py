"""Non-negotiable correctness gate: no future information leaks into features.

The decisive check is *truncation invariance*: the feature row at time t computed on
the full series must equal the feature row at t computed on the series truncated at t.
If any feature peeked at t+1, truncating the future away would change its value at t.
We also pin down that the target is strictly future and that the assembled table is
clean (no NaN, sorted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twii_forecast import data, dataset, features, target


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    df, _ = data.load(use_cache=True)
    return df


@pytest.fixture(scope="module")
def feats(ohlcv) -> pd.DataFrame:
    return features.build_features(ohlcv, volume_ok=True)


def test_features_are_causal_truncation_invariant(ohlcv, feats):
    """Truncating the series at t must not change the feature row at t."""
    full = feats
    n = len(ohlcv)
    # Sample positions across the (post-warmup) history, including the last row.
    positions = [200, 800, 1500, 3000, n - 100, n - 2, n - 1]
    for pos in positions:
        t = ohlcv.index[pos]
        truncated = features.build_features(ohlcv.iloc[: pos + 1], volume_ok=True)
        a = full.loc[t]
        b = truncated.loc[t]
        # Compare only where the full-series row is defined (post-warmup).
        defined = a.notna()
        np.testing.assert_allclose(
            a[defined].to_numpy(dtype=float),
            b[defined].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-9,
            err_msg=f"feature row at {t} changed when future was truncated -> leakage",
        )


def test_target_is_strictly_future(ohlcv):
    """y_t must equal ln(C_{t+1}/C_t); the last row is NaN (no t+1)."""
    y = target.next_day_log_return(ohlcv)
    c = ohlcv["Close"]
    expected = np.log(c.shift(-1) / c)
    pd.testing.assert_series_equal(y, expected, check_names=False)
    assert np.isnan(y.iloc[-1]), "final target must be NaN (no next day observed)"


def test_target_does_not_appear_among_features(feats):
    assert dataset.TARGET_COL not in feats.columns


def test_assembled_dataset_is_clean(ohlcv):
    table, feature_cols = dataset.build_dataset(df_ohlcv=ohlcv, volume_ok=True)
    assert not table.isna().any().any(), "assembled table must have no NaNs"
    assert table.index.is_monotonic_increasing
    assert dataset.TARGET_COL in table.columns
    assert len(feature_cols) > 0
    # Warmup rows (longest window) should have been dropped from the front.
    assert table.index[0] > ohlcv.index[0]


def test_shifting_a_feature_breaks_alignment(ohlcv, feats):
    """Sanity: a future-shifted feature would correlate differently with the target.

    Guards the intent of the truncation test — a feature carrying t+1 info (here
    simulated by shift(-1)) is genuinely different from its causal version.
    """
    y = target.next_day_log_return(ohlcv).dropna()
    causal = feats["daily_return"].reindex(y.index)
    leaked = feats["daily_return"].shift(-1).reindex(y.index)  # peeks one day ahead
    joined = pd.concat([y, causal, leaked], axis=1).dropna()
    ic_causal = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    ic_leaked = joined.iloc[:, 0].corr(joined.iloc[:, 2])
    # The leaked version (tomorrow's return vs tomorrow's target) must differ clearly.
    assert abs(ic_leaked - ic_causal) > 1e-6
