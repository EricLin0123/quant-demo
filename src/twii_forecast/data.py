"""Download ^TWII daily OHLCV via yfinance, cache to parquet, gate volume quality.

An index ticker has no real Adjusted Close (no dividend reinvestment), so we set
AdjClose := Close upstream and never carry a separate column. Yahoo's index volume
is frequently degenerate (0/NaN); `validate_volume` decides at runtime whether the
volume features are trustworthy.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download(
    ticker: str = config.TICKER,
    start: str = config.START_DATE,
    end: str | None = config.END_DATE,
) -> pd.DataFrame:
    """Pull daily OHLCV from yfinance, normalised to a flat-column DataFrame."""
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yfinance >=0.2 returns a MultiIndex (field, ticker); flatten it.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[[c for c in OHLCV if c in raw.columns]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    # An index has no dividend-adjusted close: AdjClose := Close. We simply never
    # emit a separate AC column, collapsing AC-based indicators onto Close.
    return df


def validate_volume(df: pd.DataFrame) -> bool:
    """True if the Volume column is usable (< threshold fraction zero or NaN)."""
    if "Volume" not in df.columns:
        return False
    vol = df["Volume"]
    frac_zero = (vol == 0).mean()
    frac_nan = vol.isna().mean()
    degenerate = float(frac_zero + frac_nan)
    logger.info(
        "volume gate: zero=%.3f nan=%.3f degenerate=%.3f threshold=%.3f",
        frac_zero, frac_nan, degenerate, config.VOLUME_DEGENERATE_THRESHOLD,
    )
    return degenerate < config.VOLUME_DEGENERATE_THRESHOLD


def load(
    use_cache: bool = True,
    ticker: str = config.TICKER,
    start: str = config.START_DATE,
    end: str | None = config.END_DATE,
) -> tuple[pd.DataFrame, bool]:
    """Return (ohlcv_df, volume_ok). Caches the raw pull to parquet.

    `volume_ok` is the gate result — downstream feature code uses it to decide
    whether to emit volume-derived features.
    """
    if use_cache and config.RAW_PARQUET.exists():
        df = pd.read_parquet(config.RAW_PARQUET)
        logger.info("loaded cached raw data: %d rows", len(df))
    else:
        df = download(ticker, start, end)
        df.to_parquet(config.RAW_PARQUET)
        logger.info("downloaded and cached %d rows -> %s", len(df), config.RAW_PARQUET)

    volume_ok = validate_volume(df)
    if not volume_ok:
        logger.warning(
            "volume failed quality gate; volume features will be DROPPED. "
            "(Authoritative TWSE turnover join not enabled in this build.)"
        )
    return df, volume_ok
