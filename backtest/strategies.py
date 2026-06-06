"""Strategy zoo — turning the *one* GB signal into several tradeable books.

The model (`model/train.py`) emits a single cross-sectional score, `pred`. The
long-short quintile book in `portfolio.py` is only *one* way to trade that score,
and on this universe it's the worst one: shorting a TSMC-led, decade-long bull
market with ~10 names a leg just bleeds. The signal itself is fine (mean IC
≈ 0.04, ICIR ≈ 3) — the question is portfolio *construction*.

This module reuses the **same `pred`** and the **same 50 names**, and only varies
how scores become weights. Every builder returns the rebalance-date target frame
`[date, ticker, weight]` that `engine.backtest` consumes, so they're directly
comparable on a net-of-cost basis against the 0050 benchmark.

Each strategy pulls a different, nameable lever:

  * `long_short`            — the baseline (idealized, shortable). The loser.
  * `top_quintile`          — long-only top 10, equal weight. The naive fix.
  * `top_quintile_slow`     — same, rebalanced 4× less often. Lever: turnover/cost.
  * `concentrated_top5`     — long-only top decile (~5 names). Lever: concentration.
  * `rank_tilt`             — hold all 50, weight ∝ score rank. Lever: soft tilt
                              instead of a hard cutoff (enhanced indexing).
  * `inverse_vol_quintile`  — top 10, inverse-vol weighted. A deliberate *negative*
                              result: low-vol tilting steers away from the high-vol
                              winners (TSMC et al.) that carry this index.

    uv run backtest/strategies.py   # backtests them all, prints the league table
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from backtest import engine, portfolio  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _rebal_filter(preds: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Keep only the rows on every `freq`-th trading session (the rebalances)."""
    df = preds[["date", "ticker", "pred", "sector"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    keep = set(pd.Timestamp(d) for d in portfolio.rebalance_dates(df["date"], freq))
    return df[df["date"].isin(keep)]


def _emit(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Strategy builders — each (preds, rets) -> [date, ticker, weight]            #
# --------------------------------------------------------------------------- #
def long_short(preds: pd.DataFrame, rets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Baseline: sector-neutral, dollar-neutral quintile long-short (the loser)."""
    return portfolio.build_weights(preds, long_only=False, rebalance_freq=5)


def top_quintile(preds: pd.DataFrame, rets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long-only top quintile (~10 names), equal weight, 5-day rebalance."""
    return portfolio.build_weights(preds, long_only=True, rebalance_freq=5)


def top_quintile_slow(preds: pd.DataFrame, rets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same book as `top_quintile`, rebalanced every 20 days to cut cost drag."""
    return portfolio.build_weights(preds, long_only=True, rebalance_freq=20)


def concentrated_top5(preds: pd.DataFrame, rets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long-only top *decile* (~5 names), equal weight, 10-day rebalance.

    `n_quantiles=10` makes each leg ~5 names; the per-name cap can't fit 5 names
    fully invested, so `build_weights` rescales to an equal 1/5 each — maximum
    conviction in the model's very best picks.
    """
    return portfolio.build_weights(preds, n_quantiles=10, long_only=True, rebalance_freq=10)


def rank_tilt(preds: pd.DataFrame, rets: pd.DataFrame | None = None,
              freq: int = 10) -> pd.DataFrame:
    """Enhanced-index tilt: hold *all 50*, weight ∝ cross-sectional score rank.

    No hard top/bottom cutoff — every name gets a weight proportional to where it
    ranks today (best name heaviest, worst name lightest but still > 0). It's a
    *soft* overweight of the signal on top of an equal-ish base, which keeps the
    book diversified and low-turnover while still leaning into `pred`. Fully
    invested, long-only (sums to 1.0).
    """
    df = _rebal_filter(preds, freq)
    parts = []
    for d, g in df.groupby("date"):
        r = g["pred"].rank(method="first")          # 1..n, deterministic ties
        w = r / r.sum()                              # normalize to fully invested
        parts.append(pd.DataFrame({"date": d, "ticker": g["ticker"].values,
                                   "weight": w.values}))
    return _emit(parts)


def inverse_vol_quintile(preds: pd.DataFrame, rets: pd.DataFrame,
                         freq: int = 10, vol_window: int = 21) -> pd.DataFrame:
    """Long-only top quintile, weighted by inverse trailing realized vol.

    Picks the same top-10 names as `top_quintile` but down-weights the volatile
    ones. Included as an honest *negative* control: on a momentum-driven, TSMC-led
    index the high-vol names are often the winners, so de-risking by vol tilts
    away from exactly what's carrying the benchmark.
    """
    vol = rets.rolling(vol_window).std().shift(1)    # point-in-time (≤ d-1)
    df = _rebal_filter(preds, freq)
    parts = []
    for d, g in df.groupby("date"):
        n = len(g); k = max(n // config.N_QUANTILES, 1)
        top = g.sort_values("pred").iloc[-k:]
        if d not in vol.index:
            iv = pd.Series(1.0, index=top["ticker"].values)
        else:
            v = vol.loc[d].reindex(top["ticker"].values)
            iv = (1.0 / v).replace([np.inf, -np.inf], np.nan)
            iv = iv.fillna(iv.mean())
        w = iv / iv.sum()
        parts.append(pd.DataFrame({"date": d, "ticker": w.index, "weight": w.values}))
    return _emit(parts)


# Registry: display name -> (builder, long_only?) for the comparison loop.
STRATEGIES = {
    "Long-short (idealized)": long_short,
    "Long-only top-quintile": top_quintile,
    "Top-quintile, slow (20d)": top_quintile_slow,
    "Concentrated top-5": concentrated_top5,
    "Rank-weighted tilt (all 50)": rank_tilt,
    "Inverse-vol top-quintile": inverse_vol_quintile,
}


def run_all(preds: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Backtest every strategy + the benchmark; return a sorted league table."""
    rows = []
    win = None
    for name, fn in STRATEGIES.items():
        w = fn(preds, rets)
        res = engine.backtest(w, rets)
        if win is None:
            win = res["net_ret"].index
        rows.append({
            "strategy": name,
            "net_sharpe": res["sharpe_net"],
            "gross_sharpe": res["sharpe_gross"],
            "ann_return": res["ann_return"],
            "max_drawdown": res["mdd"],
            "calmar": res["calmar"],
            "avg_turnover": res["avg_turnover"],
        })
    bench = engine.equity_stats(engine.benchmark_returns(), window=win)
    rows.append({
        "strategy": "BENCHMARK 0050.TW",
        "net_sharpe": bench["sharpe"], "gross_sharpe": bench["sharpe"],
        "ann_return": bench["ann_return"], "max_drawdown": bench["mdd"],
        "calmar": bench["calmar"], "avg_turnover": 0.0,
    })
    return pd.DataFrame(rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from backtest import metrics  # noqa: E402

    print("Strategy zoo — net-of-cost league table vs 0050\n")
    preds = metrics.load_predictions()
    rets = engine.daily_returns()
    table = run_all(preds, rets)
    with pd.option_context("display.float_format", lambda x: f"{x:+.3f}"):
        print(table.to_string(index=False))
