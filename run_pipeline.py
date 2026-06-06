"""End-to-end pipeline: data → features → model → metrics → backtest → drift.

One command reproduces every number and chart in the deck from a cold cache:

    uv run run_pipeline.py            # uses caches where present
    uv run run_pipeline.py --cold     # ignore caches, rebuild everything
    uv run run_pipeline.py --no-html  # skip the (slow) Evidently HTML

Each stage is build-or-load and immutable-cached, so re-runs are instant and
every artifact traces back to this entry point. All randomness is seeded
(`config.SEED`) for bit-reproducibility.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data import ingest  # noqa: E402
from features import alpha  # noqa: E402
from model import train  # noqa: E402
from backtest import metrics, portfolio, engine  # noqa: E402
from monitoring import drift  # noqa: E402


def _seed_everything(seed: int = config.SEED) -> None:
    """Pin every RNG we touch so the run is bit-reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _hr(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def main(cold: bool = False, make_html: bool = True) -> dict:
    _seed_everything()
    t0 = time.time()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1 — data ----------------------------------------------------------------
    _hr("1 · Data ingestion")
    prices = ingest.load_prices(force=cold)
    ingest.load_index(force=cold)
    ingest.load_benchmark(force=cold)
    print(f"  prices: {len(prices):,} rows | {prices['ticker'].nunique()} tickers "
          f"| {prices['date'].min().date()} → {prices['date'].max().date()}")

    # 2 — features ------------------------------------------------------------
    _hr("2 · Feature engineering")
    feats = alpha.load_features(force=cold)
    feature_cols = alpha.feature_columns(feats)
    print(f"  features: {len(feats):,} rows | {len(feature_cols)} cols "
          "(cross-sectionally z-scored)")

    # 3 — walk-forward model --------------------------------------------------
    _hr("3 · Purged walk-forward model")
    preds = train.load_predictions(force=cold)
    print(f"  OOS predictions: {len(preds):,} rows | {preds['date'].nunique()} dates")

    # 4 — signal quality ------------------------------------------------------
    _hr("4 · Signal quality (IC / ICIR)")
    ic = metrics.daily_ic(preds)
    ic_stats = metrics.summarize_ic(ic)
    metrics.plot_rolling_ic(ic)
    print(f"  mean IC {ic_stats['mean_ic']:+.4f} | "
          f"ICIR(overlap-adj) {ic_stats['icir_ann']:+.2f} | "
          f"t-stat {ic_stats['ic_tstat']:+.2f}")

    # 5 — portfolio + cost-aware backtest -------------------------------------
    _hr("5 · Portfolio + cost-aware backtest")
    rets = engine.daily_returns(prices)
    w_ls = portfolio.build_weights(preds, long_only=False)
    w_lo = portfolio.build_weights(preds, long_only=True)
    res_ls = engine.backtest(w_ls, rets)
    res_lo = engine.backtest(w_lo, rets)
    engine.plot_equity(res_ls, title="Long-short quintile book")
    engine.plot_equity(res_lo, path=config.REPORTS_DIR / "equity_long_only.png",
                       title="Long-only top-quintile book")
    bench = engine.equity_stats(engine.benchmark_returns(), window=res_ls["net_ret"].index)
    print(f"  long-short net Sharpe {res_ls['sharpe_net']:+.2f} | "
          f"long-only net Sharpe {res_lo['sharpe_net']:+.2f} | "
          f"benchmark 0050 Sharpe {bench['sharpe']:+.2f}")

    # 6 — drift monitoring ----------------------------------------------------
    _hr("6 · Drift monitoring (three rings)")
    raw = alpha.build_raw_features(prices)
    mon = drift.run_monitoring(predictions=preds, raw_features=raw,
                               returns=rets, make_html=make_html)
    print(f"  drifted features (PSI≥{drift.PSI_MODERATE}): "
          f"{mon['n_drifted']}/{len(mon['drift'])} | "
          f"IC trend/yr {mon['ic_decay']['ic_slope_per_year']:+.4f}")

    _summary_table(ic_stats, res_ls, res_lo, bench, mon)
    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts in {config.REPORTS_DIR}/")
    return {"ic": ic_stats, "long_short": res_ls, "long_only": res_lo,
            "benchmark": bench, "drift": mon}


def _summary_table(ic_stats, res_ls, res_lo, bench, mon) -> None:
    """The results slide: one clean table of every headline number."""
    _hr("RESULTS SUMMARY")
    print("Signal quality (out-of-sample, purged walk-forward)")
    print(f"  mean daily IC          {ic_stats['mean_ic']:>+8.4f}   (band 0.01–0.04)")
    print(f"  ICIR (overlap-adj)     {ic_stats['icir_ann']:>+8.2f}   (band 0.5–2.0)")
    print(f"  ICIR (naive √252)      {ic_stats['icir_naive']:>+8.2f}   (inflated by label overlap)")
    print(f"  IC t-stat              {ic_stats['ic_tstat']:>+8.2f}")
    print(f"  IC hit rate            {ic_stats['hit_rate']:>8.1%}")

    def col(r):
        return (f"{r['sharpe_gross']:>+8.2f}{r['sharpe_net']:>+9.2f}"
                f"{r['ann_return']:>+9.1%}{r['ann_vol']:>8.1%}"
                f"{r['mdd']:>+8.1%}{r['calmar']:>+8.2f}"
                f"{r['avg_turnover']:>9.1%}{r['hit_rate']:>8.1%}")

    print("\nBacktest                Sharpe_g  Sharpe_n   AnnRet    Vol     MDD"
          "  Calmar   Turn/rb  Hit")
    print(f"  long-short (ideal) {col(res_ls)}")
    print(f"  long-only  (real)  {col(res_lo)}")
    print(f"  0050 benchmark     {bench['sharpe']:>+17.2f}{bench['ann_return']:>+9.1%}"
          f"{bench['ann_vol']:>8.1%}{bench['mdd']:>+8.1%}{bench['calmar']:>+8.2f}"
          f"{0.0:>9.1%}{'—':>8}  (cap-weighted buy & hold)")

    d = mon["ic_decay"]
    print("\nMonitoring")
    print(f"  IC first/second half   {d['ic_first_half']:+.4f} / {d['ic_second_half']:+.4f}"
          "   (signal decay check)")
    print(f"  drifted features       {mon['n_drifted']}/{len(mon['drift'])}  "
          f"(top: {', '.join(mon['drift'].head(3)['feature'])})")
    print(f"  high-vol regime days   {mon['high_vol_frac']:.1%}")

    print(f"\nHonest read: a modest, real cross-sectional signal (IC "
          f"{ic_stats['mean_ic']:+.3f}). The long-short\nnet Sharpe is eaten by "
          "Taiwan transaction costs (0.1425%/side fee + 0.3% sell tax) on high\n"
          "turnover — the fix is a longer hold / signal smoothing, not a fancier "
          "model.\nAgainst the cap-weighted 0050 benchmark "
          f"(Sharpe {bench['sharpe']:+.2f}), the model does not\nbeat simple "
          "buy-and-hold after costs; the deliverable is the rigorous, honest loop.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold", action="store_true", help="ignore caches, rebuild all")
    ap.add_argument("--no-html", dest="html", action="store_false",
                    help="skip the Evidently HTML report (faster)")
    args = ap.parse_args()
    main(cold=args.cold, make_html=args.html)
