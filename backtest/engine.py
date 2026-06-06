"""Stage 6 — Vectorized backtester + transaction costs.

~90 lines of our own engine. No black-box backtest library on purpose: owning
the loop is what proves there's no hidden look-ahead, and the whole credibility
of the deck rests on the gross-vs-net honesty, so we don't outsource it.

The two things an engine like this must get right:

  1. **Timing (no look-ahead).** Target weights are struck from the prediction
     at the *close* of a rebalance day `t`. You can't trade on a price you used
     to decide — so those weights only earn returns from day `t+1` onward. We
     forward-fill the targets (hold between rebalances) and then `.shift(1)`, so
     the weights multiplying day `d`'s return were always set on data ≤ `d-1`.
  2. **Costs are real, and modeled the way Taiwan actually charges them.** At
     each rebalance we convert every name's weight change into an NT$ trade value
     (|Δweight| × NAV) and apply the real TW retail stack:
       * broker fee 0.1425% on *both* buy and sell, with an **NT$20 minimum per
         execution** — which is why we need an NAV, not just a turnover fraction;
       * a 0.30% **securities transaction tax on the sell side** only.
     We charge it on the day the book actually changes and report the equity
     curve **gross and net**. The drop between the two is the single most
     important chart in the deck — if net is marginal, the honest fix is lower
     turnover (longer hold / signal smoothing), not a fancier model.

    uv run backtest/engine.py   # runs the long-short book, prints the metrics table
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `uv run backtest/engine.py` to import the top-level config + siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from data import ingest  # noqa: E402

ANN = 252  # trading sessions / year, for annualization


def daily_returns(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-ticker simple daily returns as a [date × ticker] matrix.

    Built from adjusted close. Gaps were forward-filled at ingestion, so a
    filled day yields a 0% return (no fake jump) rather than a spurious move.
    """
    if prices is None:
        prices = ingest.load_prices()
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    return close.pct_change()


def benchmark_returns(bench: pd.DataFrame | None = None) -> pd.Series:
    """Daily total return of the cap-weighted benchmark ETF (0050.TW)."""
    if bench is None:
        bench = ingest.load_benchmark()
    s = bench.set_index("date")["close"].sort_index()
    return s.pct_change().rename("benchmark")


def equity_stats(ret: pd.Series, window: pd.DatetimeIndex | None = None) -> dict:
    """Buy-and-hold metrics for a return series (e.g. the benchmark).

    Restrict to `window` (the strategy's OOS dates) for an apples-to-apples
    comparison. Turnover/cost are ~nil for buy-and-hold, so we report it gross.
    """
    r = ret.dropna()
    if window is not None:
        r = r.reindex(window).dropna()
    curve = (1.0 + r).cumprod()
    ann_ret = float(r.mean() * ANN)
    return {
        "ann_return": ann_ret,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)),
        "sharpe": _sharpe(r),
        "mdd": _max_drawdown(curve),
        "calmar": float(ann_ret / abs(_max_drawdown(curve))) if _max_drawdown(curve) else float("nan"),
        "curve": curve,
        "ret": r,
    }


def backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    nav: float = config.CAPITAL_TWD,
    fee_rate: float = config.FEE_RATE,
    min_fee: float = config.MIN_FEE_TWD,
    sell_tax: float = config.SELL_TAX_RATE,
) -> dict:
    """Run the weight schedule against daily returns; return curves + metrics.

    `weights` is the rebalance-date target frame [date, ticker, weight] from
    `portfolio.build_weights`; `returns` is the [date × ticker] daily-return
    matrix. Costs use the explicit Taiwan stack (see module docstring): a
    per-name broker fee with an NT$`min_fee` minimum, plus a sell-side tax, all
    scaled by `nav`. Returns a dict with the gross/net equity curves, per-day
    returns, turnover, and the headline stats.
    """
    # Targets as a [rebalance_date × ticker] matrix on the traded universe.
    # Crucially, fill 0 here: a name absent from a rebalance means "not held = 0"
    # on that date. If we left it NaN, the ffill below would carry a *stale*
    # weight from an earlier rebalance forward, silently accumulating positions.
    target = (weights.pivot(index="date", columns="ticker", values="weight")
              .reindex(columns=returns.columns)
              .fillna(0.0))

    # Trade only inside the OOS window; align the return matrix to those days.
    days = returns.index[(returns.index >= target.index.min())
                         & (returns.index <= returns.index.max())]
    held = target.reindex(days).ffill().fillna(0.0)   # hold between rebalances
    active = held.shift(1).fillna(0.0)                 # weights set on data ≤ d-1
    rets = returns.reindex(days).fillna(0.0)[active.columns]

    # Gross daily P&L = Σ weightᵢ · returnᵢ.
    gross_ret = (active * rets).sum(axis=1)

    # Per-name weight change at each session (non-zero only the day after each
    # rebalance). The first row is the initial build from cash.
    dW = active.diff()
    dW.iloc[0] = active.iloc[0]
    traded = dW.abs()                                  # |Δw| per name
    sold = (-dW).clip(lower=0.0)                        # weight *reductions* = sells

    # Broker fee per name = max(fee_rate · NT$ trade value, NT$ minimum), but only
    # where a trade actually happened. Sell tax hits the sold notional only.
    trade_twd = traded * nav
    fee_twd = np.maximum(fee_rate * trade_twd, min_fee).where(traded > 0, 0.0)
    tax_twd = sell_tax * sold * nav
    cost = (fee_twd.sum(axis=1) + tax_twd.sum(axis=1)) / nav   # back to a return

    turnover = traded.sum(axis=1)                      # Σ|Δw|, for reporting
    net_ret = gross_ret - cost

    gross_curve = (1.0 + gross_ret).cumprod()
    net_curve = (1.0 + net_ret).cumprod()
    rebal_turnover = turnover[turnover > 0]            # one entry per rebalance

    return {
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "gross_curve": gross_curve,
        "net_curve": net_curve,
        "turnover": turnover,
        "cost": cost,
        **_stats(gross_ret, net_ret, net_curve, rebal_turnover),
    }


def _sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd else float("nan")


def _max_drawdown(curve: pd.Series) -> float:
    """Most negative peak-to-trough drop of an equity curve (≤ 0)."""
    return float((curve / curve.cummax() - 1.0).min())


def _stats(gross_ret, net_ret, net_curve, rebal_turnover) -> dict:
    """Headline performance metrics, computed on the **net** series."""
    ann_ret = float(net_ret.mean() * ANN)
    ann_vol = float(net_ret.std(ddof=1) * np.sqrt(ANN))
    mdd = _max_drawdown(net_curve)
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_gross": _sharpe(gross_ret),
        "sharpe_net": _sharpe(net_ret),
        "mdd": mdd,
        "calmar": float(ann_ret / abs(mdd)) if mdd else float("nan"),
        "avg_turnover": float(rebal_turnover.mean()),    # per rebalance, Σ|Δw|
        "hit_rate": float((net_ret > 0).mean()),         # share of up days, net
        "cost_drag": float((gross_ret.mean() - net_ret.mean()) * ANN),
    }


def plot_equity(result: dict, path: Path = config.REPORTS_DIR / "equity_gross_net.png",
                title: str = "Long-short quintile book") -> Path:
    """The deck's money chart: gross vs net equity. The gap *is* the cost story."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g, n = result["gross_curve"], result["net_curve"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(g.index, g.values, color="C0", lw=1.6, label="gross")
    ax.plot(n.index, n.values, color="C3", lw=1.6, label="net of costs")
    ax.fill_between(n.index, n.values, g.values, color="C3", alpha=0.10)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_title(f"{title} — gross vs net equity "
                 f"(net Sharpe {result['sharpe_net']:+.2f})")
    ax.set_ylabel("growth of $1"); ax.set_xlabel("date")
    ax.legend(loc="upper left"); ax.margins(x=0.01)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Acceptance checks                                                            #
# --------------------------------------------------------------------------- #
def _report(name: str, res: dict) -> None:
    print(f"\n{name}")
    print(f"  ann return (net)   {res['ann_return']:>+8.2%}")
    print(f"  ann vol            {res['ann_vol']:>8.2%}")
    print(f"  Sharpe  gross/net  {res['sharpe_gross']:>+8.2f} / {res['sharpe_net']:+.2f}")
    print(f"  max drawdown       {res['mdd']:>+8.2%}")
    print(f"  Calmar             {res['calmar']:>+8.2f}")
    print(f"  avg turnover/rebal {res['avg_turnover']:>8.1%}  (Σ|Δw| per rebalance)")
    print(f"  cost drag (ann)    {res['cost_drag']:>+8.2%}")
    print(f"  hit rate (net)     {res['hit_rate']:>8.1%}")


def _acceptance(res: dict) -> None:
    """plan §6: net strictly below gross; turnover realistic (not 300%/day)."""
    g, n = res["gross_curve"], res["net_curve"]
    assert (n <= g + 1e-12).all(), "net curve rises above gross — cost sign bug"
    assert n.iloc[-1] < g.iloc[-1], "net did not end below gross — no cost applied?"
    # avg_turnover is Σ|Δw| per rebalance (every REBALANCE_FREQ days). A daily
    # equivalent of ~300%/day would be absurd; ours should be a fraction of 1.
    daily_equiv = res["avg_turnover"] / config.REBALANCE_FREQ
    assert daily_equiv < 1.0, f"turnover {daily_equiv:.0%}/day is unrealistic"
    print(f"\n  ✓ net strictly below gross; turnover ≈ {daily_equiv:.1%}/day "
          f"({res['avg_turnover']:.1%} per {config.REBALANCE_FREQ}-day rebalance) — realistic.")


if __name__ == "__main__":
    from backtest import metrics, portfolio  # noqa: E402

    print("Stage 6 — vectorized backtest + costs\n")
    preds = metrics.load_predictions()
    rets = daily_returns()

    w_ls = portfolio.build_weights(preds)
    res_ls = backtest(w_ls, rets)
    _report("Long-short, sector-neutral (idealized — assumes shortable):", res_ls)
    _acceptance(res_ls)
    out = plot_equity(res_ls)
    print(f"  equity chart -> {out}")

    w_lo = portfolio.build_weights(preds, long_only=True)
    res_lo = backtest(w_lo, rets)
    _report("Long-only top quintile (realistic, shortable deployment):", res_lo)
    plot_equity(res_lo, path=config.REPORTS_DIR / "equity_long_only.png",
                title="Long-only top-quintile book")

    bench = equity_stats(benchmark_returns(), window=res_ls["net_ret"].index)
    print(f"\nBenchmark 0050.TW (cap-weighted top-50, buy & hold): "
          f"Sharpe {bench['sharpe']:+.2f} | annRet {bench['ann_return']:+.1%} "
          f"| vol {bench['ann_vol']:.1%} | MDD {bench['mdd']:+.1%}")

    print("\nStage 6 done.")
