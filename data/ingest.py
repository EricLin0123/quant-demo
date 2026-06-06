"""Stage 1 — Data ingestion.

Pull adjusted daily OHLCV for the TWSE top-50 universe (plus the TAIEX index
proxy) via yfinance, align everyone to a common trading calendar, and cache to
parquet. The raw pull is treated as **immutable**: once `prices.parquet` exists
we read it back instead of re-hitting the network.

Run directly to (re)build the cache:

    uv run data/ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Allow `uv run data/ingest.py` (script dir on path) to import top-level config.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLCV and return a tidy long frame.

    Columns: [date, ticker, open, high, low, close, volume].
    `auto_adjust=True` folds splits/dividends into OHLC so returns are clean.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        progress=True,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data — check tickers / network.")

    frames = []
    # Single ticker comes back with flat columns; multi-ticker is a column
    # MultiIndex keyed by ticker. Normalise both into the same long shape.
    multi = isinstance(raw.columns, pd.MultiIndex)
    for tkr in tickers:
        sub = raw[tkr] if multi else raw
        sub = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
        sub.columns = ["open", "high", "low", "close", "volume"]
        sub = sub.dropna(how="all")
        if sub.empty:
            print(f"  ! no data for {tkr}, skipping")
            continue
        sub.insert(0, "ticker", tkr)
        frames.append(sub.reset_index().rename(columns={"Date": "date"}))

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _align_calendar(df: pd.DataFrame, max_missing_frac: float) -> pd.DataFrame:
    """Reindex every ticker onto the common trading calendar.

    The calendar is the union of all observed trading dates. Names missing more
    than `max_missing_frac` of those days (e.g. listed partway through) are
    dropped. Remaining gaps in OHLC are forward-filled — but we NEVER fill the
    forward-return target (that lives in Stage 2 and must stay honest).
    """
    calendar = pd.Index(sorted(df["date"].unique()), name="date")
    n_days = len(calendar)

    kept, dropped = [], []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.set_index("date").reindex(calendar)
        missing_frac = g["close"].isna().mean()
        if missing_frac > max_missing_frac:
            dropped.append((tkr, missing_frac))
            continue
        # Forward-fill prices carefully; back-fill the leading edge so the very
        # first rows aren't NaN. Volume gaps -> 0 (no trades that session).
        price_cols = ["open", "high", "low", "close"]
        g[price_cols] = g[price_cols].ffill().bfill()
        g["volume"] = g["volume"].fillna(0.0)
        g["ticker"] = tkr
        kept.append(g.reset_index())

    if dropped:
        print(f"  dropped {len(dropped)} ticker(s) over "
              f"{max_missing_frac:.0%} missing: "
              + ", ".join(f"{t}({f:.0%})" for t, f in dropped))

    out = pd.concat(kept, ignore_index=True)
    print(f"  aligned {out['ticker'].nunique()} tickers x {n_days} trading days")
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]


def load_prices(
    tickers: list[str] | None = None,
    start: str = config.START,
    end: str = config.END,
    cache: Path = config.PRICES_CACHE,
    force: bool = False,
) -> pd.DataFrame:
    """Load the universe price panel, using the parquet cache when present.

    Returns a tidy long frame: [date, ticker, open, high, low, close, volume].
    The cache is immutable: if it exists and `force` is False we read it back.
    """
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    tickers = tickers or config.TWSE_TOP50
    cache.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pulling {len(tickers)} tickers {start} -> {end} ...")
    df = _download(tickers, start, end)
    df = _align_calendar(df, config.MAX_MISSING_FRAC)

    df.to_parquet(cache, index=False)
    print(f"  cached -> {cache}")
    return df


def load_index(
    symbol: str = config.INDEX_PROXY,
    start: str = config.START,
    end: str = config.END,
    cache: Path = config.INDEX_CACHE,
    force: bool = False,
) -> pd.DataFrame:
    """Load the index proxy (TAIEX) for the later beta feature. [date, close]."""
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pulling index {symbol} {start} -> {end} ...")
    df = _download([symbol], start, end)[["date", "close"]]
    df.to_parquet(cache, index=False)
    print(f"  cached -> {cache}")
    return df


def _sanity_check(prices: pd.DataFrame) -> None:
    """Cheap assertions so a bad pull fails loudly instead of silently."""
    assert not prices[["date", "ticker"]].duplicated().any(), "duplicate (date,ticker) rows"
    assert (prices["close"] > 0).all(), "non-positive close prices present"
    span = prices.groupby("ticker")["date"].agg(["min", "max", "count"])
    print("\nPer-ticker coverage (head):")
    print(span.head(8).to_string())
    print(f"\nTotal rows: {len(prices):,} | "
          f"tickers: {prices['ticker'].nunique()} | "
          f"dates: {prices['date'].min().date()} -> {prices['date'].max().date()}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    prices = load_prices(force=force)
    index = load_index(force=force)
    _sanity_check(prices)
    print(f"\nIndex rows: {len(index):,} "
          f"({index['date'].min().date()} -> {index['date'].max().date()})")
    print("\nStage 1 done.")
