"""Causal technical indicators.

Every rolling operation here is **trailing** (``min_periods == window``, never
centered): the value at time ``t`` depends only on information available at or
before ``t``. That is what makes it safe to compute features on the full series
before the chronological split — there is no future bleed. ``tests/test_leakage``
enforces this property.

With ``AdjClose := Close`` for an index, every AC-based indicator collapses onto
its Close equivalent, so we compute each once and skip the duplicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Columns that are derived from Volume; dropped if the volume gate fails.
VOLUME_FEATURES: list[str] = []


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI on close, trailing only."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing via EMA with alpha = 1/window (uses only past values).
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    hh = high.rolling(window, min_periods=window).max()
    ll = low.rolling(window, min_periods=window).min()
    return (hh - close) / (hh - ll) * -100


def build_features(df: pd.DataFrame, use_volume: bool = True) -> pd.DataFrame:
    """Return a feature matrix indexed like ``df``. Warmup NaNs kept (dropped later)."""
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    prev_c = c.shift(1)

    feat = pd.DataFrame(index=df.index)

    # --- spreads / single-bar shape -----------------------------------------
    feat["amplitude"] = (h - l) / prev_c
    feat["difference"] = (c - o) / prev_c          # paper's top-SHAP feature
    feat["h_minus_l"] = h - l
    feat["c_minus_o"] = c - o

    # --- first differences ---------------------------------------------------
    dC = c.diff()
    feat["delta_close"] = dC
    if use_volume:
        dV = v.diff()
        feat["delta_volume"] = dV

    # --- intraday term: I = O_{t-1} - C_{t-1} (causal, all lagged) -----------
    intraday = o.shift(1) - c.shift(1)
    feat["intraday"] = intraday

    # --- short-window moving averages of intraday / dC / dV -----------------
    for n in config.MA_SHORT_WINDOWS:
        feat[f"intraday_ma{n}"] = intraday.rolling(n, min_periods=n).mean()
        feat[f"delta_close_ma{n}"] = dC.rolling(n, min_periods=n).mean()
        if use_volume:
            feat[f"delta_volume_ma{n}"] = dV.rolling(n, min_periods=n).mean()

    # --- direction of open ---------------------------------------------------
    feat["sign_delta_open"] = np.sign(o.diff()).fillna(0.0)

    # --- returns / momentum --------------------------------------------------
    daily_ret = np.log(c / prev_c)
    feat["daily_return"] = daily_ret
    feat["momentum"] = c - c.shift(config.MOMENTUM_LAG)

    # --- oscillators ---------------------------------------------------------
    feat["rsi"] = _rsi(c, config.RSI_WINDOW)
    feat["williams_r"] = _williams_r(h, l, c, config.WILLIAMS_WINDOW)

    # --- trend: MA / EMA on close -------------------------------------------
    for n in config.MA_PRICE_WINDOWS:
        feat[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
        feat[f"ema{n}"] = c.ewm(span=n, min_periods=n, adjust=False).mean()

    # --- MACD ----------------------------------------------------------------
    ema_fast = c.ewm(span=config.MACD_FAST, min_periods=config.MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=config.MACD_SLOW, min_periods=config.MACD_SLOW, adjust=False).mean()
    feat["macd"] = ema_fast - ema_slow

    # --- Bollinger Bands -----------------------------------------------------
    roll = c.rolling(config.BB_WINDOW, min_periods=config.BB_WINDOW)
    bb_mid, bb_std = roll.mean(), roll.std()
    feat["bb_high"] = bb_mid + 2 * bb_std
    feat["bb_low"] = bb_mid - 2 * bb_std

    # --- lagged returns ------------------------------------------------------
    for k in config.LAG_RETURNS:
        feat[f"ret_lag{k}"] = daily_ret.shift(k)

    # Record which columns are volume-derived (for documentation / drift split).
    global VOLUME_FEATURES
    VOLUME_FEATURES = [col for col in feat.columns if "volume" in col]

    return feat
