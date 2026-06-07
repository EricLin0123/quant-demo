"""Technical indicators — every column is strictly causal (trailing only).

Leakage rule: the feature row at time ``t`` depends only on observations at times
``<= t``. Every rolling op uses ``min_periods = window`` (never centred); EMAs use
``adjust=False`` so they are pure trailing recursions. This is what lets us compute
features on the full series *before* splitting without leaking the future, and is
exactly what ``tests/test_leakage.py`` checks by truncating the series at ``t``.

With ``AdjClose := Close`` the AC-based indicators collapse onto their Close
equivalents, so we never emit the duplicate columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder-style RSI, computed causally with trailing rolling means."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 (all gains) RSI -> 100; when avg_gain == 0 -> 0.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    return rsi


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Williams %R in [-100, 0], trailing window."""
    hh = high.rolling(window, min_periods=window).max()
    ll = low.rolling(window, min_periods=window).min()
    rng = (hh - ll).replace(0.0, np.nan)
    return (hh - close) / rng * -100.0


def build_features(df: pd.DataFrame, volume_ok: bool = True) -> pd.DataFrame:
    """Return a DataFrame of causal technical features indexed by Date.

    `volume_ok` toggles the volume-derived block (per the runtime data-quality gate).
    NaN warmup rows from the longest window are dropped by the caller (`dataset.py`).
    """
    o, h, l, c, v = (df["Open"], df["High"], df["Low"], df["Close"], df.get("Volume"))
    prev_c = c.shift(1)

    feats: dict[str, pd.Series] = {}

    # --- price-shape / intraday ------------------------------------------- #
    feats["amplitude"] = (h - l) / prev_c
    feats["difference"] = (c - o) / prev_c          # paper's top SHAP feature
    feats["h_minus_l"] = h - l
    feats["c_minus_o"] = c - o

    # --- first differences ------------------------------------------------- #
    delta_c = c.diff()
    feats["delta_c"] = delta_c
    intraday = c - o                                # intraday move, basis for "Intraday MA"

    # --- daily return / momentum ------------------------------------------ #
    daily_ret = np.log(c / prev_c)
    feats["daily_return"] = daily_ret
    feats["momentum"] = c - c.shift(config.MOMENTUM_WINDOW)
    feats["sign_delta_open"] = np.sign(o - o.shift(1))

    # --- short-window deltas of intraday / dC (and dV when usable) -------- #
    for n in config.INTRADAY_MA_WINDOWS:
        feats[f"intraday_ma_{n}"] = intraday.rolling(n, min_periods=n).mean()
        feats[f"delta_c_ma_{n}"] = delta_c.rolling(n, min_periods=n).mean()

    # --- moving averages / EMAs of close ---------------------------------- #
    for n in config.MA_WINDOWS:
        feats[f"ma_{n}"] = c.rolling(n, min_periods=n).mean()
        feats[f"ema_{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()

    # --- MACD -------------------------------------------------------------- #
    ema_fast = c.ewm(span=config.MACD_FAST, adjust=False, min_periods=config.MACD_FAST).mean()
    ema_slow = c.ewm(span=config.MACD_SLOW, adjust=False, min_periods=config.MACD_SLOW).mean()
    feats["macd"] = ema_fast - ema_slow

    # --- oscillators ------------------------------------------------------- #
    feats["rsi"] = _rsi(c, config.RSI_WINDOW)
    feats["williams_r"] = _williams_r(h, l, c, config.WILLIAMS_WINDOW)

    # --- Bollinger bands --------------------------------------------------- #
    mid = c.rolling(config.BB_WINDOW, min_periods=config.BB_WINDOW).mean()
    sd = c.rolling(config.BB_WINDOW, min_periods=config.BB_WINDOW).std()
    feats["bb_high"] = mid + config.BB_STD * sd
    feats["bb_low"] = mid - config.BB_STD * sd

    # --- lagged returns ---------------------------------------------------- #
    for k in config.LAG_RETURNS:
        feats[f"ret_lag_{k}"] = daily_ret.shift(k)

    # --- volume block (only if the runtime gate passed) ------------------- #
    if volume_ok and v is not None:
        delta_v = v.diff()
        feats["delta_v"] = delta_v
        for n in config.INTRADAY_MA_WINDOWS:
            feats[f"delta_v_ma_{n}"] = delta_v.rolling(n, min_periods=n).mean()

    out = pd.DataFrame(feats, index=df.index)
    return out


def feature_columns(df_features: pd.DataFrame) -> list[str]:
    """The ordered list of feature column names."""
    return list(df_features.columns)
