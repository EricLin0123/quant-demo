"""Download ^TWII OHLCV, run the volume data-quality gate, cache to parquet.

An index ticker on Yahoo has two quirks we handle explicitly here rather than
assume away:

1. ``Volume`` is frequently 0 / NaN for ``^TWII`` -> a runtime gate decides
   whether volume features are trustworthy.
2. There is no genuine Adjusted Close for an index (no dividend reinvestment),
   so we set ``AdjClose := Close`` and drop the duplicate downstream.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from . import config

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download(force: bool = False) -> pd.DataFrame:
    """Return cached OHLCV if present, otherwise pull from Yahoo and cache it."""
    if config.RAW_PARQUET.exists() and not force:
        logger.info("Loading cached raw data from %s", config.RAW_PARQUET)
        return pd.read_parquet(config.RAW_PARQUET)

    logger.info("Downloading %s from Yahoo Finance", config.TICKER)
    raw = yf.download(
        config.TICKER,
        start=config.BENCHMARK_START,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {config.TICKER}")

    # yfinance can return MultiIndex columns ((field, ticker)); flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename_axis("Date").sort_index()

    # No real Adjusted Close for an index: AC := C. Keep only the canonical set.
    df = raw[OHLCV].copy()
    df.index = pd.to_datetime(df.index)

    df.to_parquet(config.RAW_PARQUET)
    logger.info("Cached %d rows -> %s", len(df), config.RAW_PARQUET)
    return df


def validate_volume(df: pd.DataFrame) -> bool:
    """True if volume is usable: <5% of rows are zero or NaN."""
    frac_zero = (df["Volume"] == 0).mean()
    frac_nan = df["Volume"].isna().mean()
    degenerate = float(frac_zero + frac_nan)
    logger.info(
        "Volume gate: zero=%.3f nan=%.3f degenerate=%.3f (threshold %.2f)",
        frac_zero, frac_nan, degenerate, config.VOLUME_DEGENERATE_THRESHOLD,
    )
    return degenerate < config.VOLUME_DEGENERATE_THRESHOLD


def load(force: bool = False) -> tuple[pd.DataFrame, bool]:
    """Download (or load cache) and report whether volume features are usable.

    Returns ``(df, use_volume)``. When ``use_volume`` is False the caller drops
    the volume-derived features rather than feed the model noise.
    """
    df = download(force=force)
    use_volume = validate_volume(df)
    if not use_volume:
        logger.warning(
            "Volume failed the data-quality gate -> volume features will be "
            "dropped. (TWSE OpenAPI turnover join is the documented fallback "
            "if real volume is required.)"
        )
    return df, use_volume
