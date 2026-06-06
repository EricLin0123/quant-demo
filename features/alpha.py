"""Stage 2 — Feature / alpha engineering.

Turn the raw price panel from Stage 1 into a model-ready feature frame:

  1. Per-stock, point-in-time time-series features (momentum, reversal,
     volatility, liquidity, trend, beta) — every value at row `t` uses only
     data observed at or before `t`.
  2. A forward-return target over `LABEL_HORIZON`, plus its cross-sectional
     rank within each date.
  3. Cross-sectional normalization: each day, z-score every feature across all
     names. This is the conceptual core — it reframes an absolute feature as
     "how this stock looks vs its peers *today*", which is what a
     dollar-neutral, cross-sectional long-short model actually trades on.

Output frame: [date, ticker, sector, <feature columns>, fwd_ret, fwd_rank].

Run directly to (re)build the feature cache:

    uv run features/alpha.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `uv run features/alpha.py` to import the top-level config + ingest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import ingest  # noqa: E402

# The raw inputs columns we never feed to the model directly.
_RAW_COLS = ["open", "high", "low", "close", "volume", "log_ret", "dollar_vol"]

# Longest lookback any feature uses; rows before this per ticker are pure warmup
# (NaN) and get dropped. 252 ≈ one trading year for the 12-1 momentum window.
WARMUP = 252


# --------------------------------------------------------------------------- #
# Per-stock time-series features                                              #
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI on a single ticker's close series (uses only past data)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _per_stock_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute time-series features for ONE ticker (already date-sorted).

    Every feature is point-in-time: rolling windows and positive `shift`s look
    only backward. Nothing here touches the future — that is the target's job.
    """
    close = g["close"]
    ret = g["log_ret"]
    dvol = g["dollar_vol"]

    feats = pd.DataFrame(index=g.index)

    # --- momentum / reversal ------------------------------------------------
    # 12-1 momentum: 12-month return skipping the most recent month (~21d) to
    # avoid short-term-reversal contamination of the momentum signal.
    feats["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    feats["mom_6_1"] = close.shift(21) / close.shift(126) - 1.0
    feats["rev_5"] = close / close.shift(5) - 1.0        # short-term reversal
    feats["rev_21"] = close / close.shift(21) - 1.0      # 1-month reversal

    # --- realized volatility ------------------------------------------------
    feats["vol_21"] = ret.rolling(21).std()
    feats["vol_63"] = ret.rolling(63).std()
    feats["vol_ratio"] = feats["vol_21"] / feats["vol_63"]   # vol regime shift

    # --- liquidity / dollar-volume -----------------------------------------
    feats["dollar_vol_21"] = np.log(dvol.rolling(21).mean() + 1.0)
    feats["dollar_vol_trend"] = (
        dvol.rolling(21).mean() / dvol.rolling(63).mean() - 1.0
    )
    # Amihud illiquidity: average |return| per dollar traded (scaled up).
    feats["amihud_21"] = (ret.abs() / (dvol + 1.0)).rolling(21).mean() * 1e9

    # --- distance from moving averages (trend) ------------------------------
    feats["dist_sma20"] = close / close.rolling(20).mean() - 1.0
    feats["dist_sma50"] = close / close.rolling(50).mean() - 1.0
    feats["dist_sma200"] = close / close.rolling(200).mean() - 1.0

    # --- shape / tails ------------------------------------------------------
    feats["rsi_14"] = _rsi(close, 14)
    feats["max_ret_21"] = ret.rolling(21).max()          # "lottery" proxy
    feats["ret_skew_63"] = ret.rolling(63).skew()
    feats["range_21"] = ((g["high"] - g["low"]) / close).rolling(21).mean()

    return feats


def _rolling_beta(prices: pd.DataFrame, index: pd.DataFrame, window: int = 63) -> pd.Series:
    """Rolling beta of each ticker's return to the index proxy (TAIEX).

    beta_t = Cov(r_stock, r_idx) / Var(r_idx) over the trailing `window` days,
    computed per ticker. Index returns are merged on date so the covariance is
    aligned. Point-in-time: only trailing data feeds each estimate.
    """
    idx = index.sort_values("date").copy()
    idx["idx_ret"] = np.log(idx["close"] / idx["close"].shift(1))
    idx = idx[["date", "idx_ret"]]

    df = prices.merge(idx, on="date", how="left").sort_values(["ticker", "date"])

    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["log_ret"].rolling(window).cov(g["idx_ret"])
        var = g["idx_ret"].rolling(window).var()
        return cov / var

    beta = df.groupby("ticker", group_keys=False).apply(_beta)
    beta.index = df.index
    return beta.reindex(prices.index)


# --------------------------------------------------------------------------- #
# Target                                                                       #
# --------------------------------------------------------------------------- #
def add_forward_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach the forward return and its cross-sectional rank.

    `fwd_ret` at row t is the simple return from close[t] to close[t+horizon],
    computed per ticker (strictly future, no overlap with the features). The
    last `horizon` rows per ticker have no future to label them with and are
    left NaN here (dropped downstream).

    `fwd_rank` is `fwd_ret` ranked within each date and scaled to [-1, 1] — a
    distribution-robust target for the cross-sectional model.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    fwd = df.groupby("ticker")["close"].shift(-horizon)
    df["fwd_ret"] = fwd / df["close"] - 1.0

    # Cross-sectional rank within each date, mapped to [-1, 1].
    r = df.groupby("date")["fwd_ret"].rank(pct=True)
    df["fwd_rank"] = 2.0 * r - 1.0
    return df


# --------------------------------------------------------------------------- #
# Cross-sectional normalization                                                #
# --------------------------------------------------------------------------- #
def _cross_sectional_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Z-score each feature across all names *within each date*.

    Each day, every feature is winsorized to its 1st/99th cross-sectional
    percentile (so one blown-up name can't dominate) and *then* standardized,
    which keeps the per-date mean exactly 0 and std 1. Days with too few valid
    names for a feature leave NaN (handled when we drop incomplete rows).
    """
    def _z(block: pd.DataFrame) -> pd.DataFrame:
        lo, hi = block.quantile(0.01), block.quantile(0.99)
        b = block.clip(lower=lo, upper=hi, axis=1)
        return (b - b.mean()) / b.std(ddof=0).replace(0.0, np.nan)

    df[feature_cols] = (
        df.groupby("date", group_keys=False)[feature_cols].apply(_z)
    )
    return df


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def build_features(
    prices: pd.DataFrame | None = None,
    index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full Stage 2 feature frame from the price panel.

    Returns [date, ticker, sector, <feature cols>, fwd_ret, fwd_rank], with all
    features cross-sectionally z-scored and warmup/unlabelled rows removed.
    """
    prices = ingest.load_prices() if prices is None else prices
    index = ingest.load_index() if index is None else index

    df = prices.sort_values(["ticker", "date"]).reset_index(drop=True).copy()

    # Derived per-row quantities the features build on.
    df["log_ret"] = df.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    df["dollar_vol"] = df["close"] * df["volume"]

    # Per-stock time-series features.
    feats = df.groupby("ticker", group_keys=False).apply(_per_stock_features)
    df = pd.concat([df, feats], axis=1)
    df["beta_63"] = _rolling_beta(df, index)

    feature_cols = [c for c in feats.columns] + ["beta_63"]

    # Sector tag + forward target.
    df["sector"] = df["ticker"].map(config.SECTORS)
    df = add_forward_target(df, config.LABEL_HORIZON)

    # Drop warmup rows (insufficient lookback) and unlabelled tail rows.
    row_in_ticker = df.groupby("ticker").cumcount()
    df = df[row_in_ticker >= WARMUP]
    df = df.dropna(subset=feature_cols + ["fwd_ret"]).reset_index(drop=True)

    # Cross-sectional normalization (the conceptual core).
    df = _cross_sectional_zscore(df, feature_cols)

    # Re-drop any rows a thin cross-section left NaN after z-scoring.
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    out_cols = ["date", "ticker", "sector"] + feature_cols + ["fwd_ret", "fwd_rank"]
    return df[out_cols].sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input columns of a built feature frame (excludes ids/target)."""
    drop = {"date", "ticker", "sector", "fwd_ret", "fwd_rank"}
    return [c for c in df.columns if c not in drop]


def load_features(cache: Path = config.FEATURES_CACHE, force: bool = False) -> pd.DataFrame:
    """Build-or-load the feature frame. Cache is immutable unless `force`."""
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    df = build_features()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"  cached -> {cache}")
    return df


# --------------------------------------------------------------------------- #
# Sanity checks                                                                #
# --------------------------------------------------------------------------- #
def _sanity_check(df: pd.DataFrame) -> None:
    """Assert the two plan acceptance criteria, loudly."""
    fcols = feature_columns(df)
    print(f"\nFeature frame: {len(df):,} rows | "
          f"{len(fcols)} features | "
          f"{df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print("Features:", ", ".join(fcols))

    assert not df[["date", "ticker"]].duplicated().any(), "duplicate (date,ticker)"
    assert df[fcols].isna().sum().sum() == 0, "NaNs remain in feature columns"

    # Acceptance: on any single date each feature has mean ≈ 0 across names.
    daily_mean = df.groupby("date")[fcols].mean()
    worst = daily_mean.abs().max().max()
    print(f"\nMax |per-date feature mean| across all days/feats: {worst:.3e} "
          f"(should be ≈ 0 — cross-sectional z-score)")
    assert worst < 1e-6, "cross-sectional features are not mean-zero per date"

    # Acceptance: forward target really is forward — spot-check one name by hand.
    tkr = df["ticker"].iloc[0]
    g = df[df["ticker"] == tkr].head(3)
    print(f"\nLeakage eyeball for {tkr} (fwd_ret should match {config.LABEL_HORIZON}d-ahead move):")
    print(g[["date", "fwd_ret", "fwd_rank"]].to_string(index=False))

    # fwd_rank is per-date uniform in [-1, 1].
    assert df["fwd_rank"].between(-1.0, 1.0).all(), "fwd_rank out of [-1,1]"


if __name__ == "__main__":
    force = "--force" in sys.argv
    df = load_features(force=force)
    _sanity_check(df)
    print("\nStage 2 done.")
