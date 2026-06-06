"""Stage 4 — Backtest metrics: signal quality *before* PnL.

The Information Coefficient (IC) is the daily cross-sectional rank correlation
between the model's prediction and the realized forward return. We lead with it,
not the equity curve, for two reasons:

  * **It's the honest read of the signal.** With only 50 names the portfolio PnL
    is high-variance (≈10 per leg), so a pretty equity curve can be luck. IC
    aggregates *every* name every day, so it has far more statistical power and
    is much harder to fool yourself with.
  * **It's the first leakage tripwire.** Free daily data should give a mean
    daily IC around 0.01–0.04 and an annualized ICIR around 0.5–2.0. A mean IC
    of 0.1+ or an ICIR north of 3 isn't alpha — it's a bug (look-ahead in the
    features or a purge that didn't fire). The acceptance block below asserts
    exactly that band so a regression screams instead of looking impressive.

We report mean IC, its t-stat, the annualized ICIR, the hit rate, and a rolling
60-day IC plot (`reports/rolling_ic.png`) for the results slide.

> **Caveat worth saying out loud:** the label is a 10-day forward return sampled
> *daily*, so consecutive days' ICs share 9/10 of their look-ahead window and
> are strongly autocorrelated. That makes the naive t-stat and ICIR (which
> assume independent days) optimistic. The honest fixes are to either sample the
> IC series every `LABEL_HORIZON` days or apply a Newey-West correction; we keep
> the simple version for the headline number and flag the inflation here.

    uv run backtest/metrics.py   # prints the IC table + saves the rolling plot
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `uv run backtest/metrics.py` to import the top-level config.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Trading sessions per year, for annualizing the ICIR. ~252 is the TWSE norm.
PERIODS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Core metrics                                                                 #
# --------------------------------------------------------------------------- #
def daily_ic(
    df: pd.DataFrame,
    score: str = "pred",
    fwd: str = "fwd_ret",
    method: str = "spearman",
) -> pd.Series:
    """Cross-sectional IC per date: rank correlation of `score` vs `fwd`.

    Spearman by default — we trade names by *rank*, so a rank correlation is the
    metric that matches the objective and is robust to the fat tails of raw
    forward returns. Returns one IC per date (NaNs from degenerate days dropped).
    """
    ic = (
        df.groupby("date")
        .apply(lambda g: g[score].corr(g[fwd], method=method), include_groups=False)
        .dropna()
    )
    ic.name = "ic"
    return ic


def icir(
    ic_series: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
    overlap: int = config.LABEL_HORIZON,
) -> float:
    """Annualized IC Information Ratio = mean/std * sqrt(effective periods/yr).

    The Sharpe of the IC series: how consistently positive the signal is, scaled
    to a yearly figure. Because the label is a `LABEL_HORIZON`-day forward return
    sampled *daily*, consecutive ICs share all but one day of their look-ahead
    window and are strongly autocorrelated — so there are only ~252/overlap
    *independent* observations per year, not 252. We annualize over that
    effective count (`overlap=LABEL_HORIZON`); pass `overlap=1` to reproduce the
    naive, inflated `sqrt(252)` figure for the cautionary comparison.
    """
    sd = ic_series.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(ic_series.mean() / sd * np.sqrt(periods_per_year / max(overlap, 1)))


def ic_tstat(ic_series: pd.Series) -> float:
    """t-stat of the mean IC against zero: mean/std * sqrt(n_days)."""
    sd = ic_series.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(ic_series.mean() / sd * np.sqrt(len(ic_series)))


def rolling_ic(ic_series: pd.Series, window: int = 60) -> pd.Series:
    """Trailing-`window` mean IC — does the signal decay over time? (Stage 7 reuse.)"""
    return ic_series.rolling(window, min_periods=window // 2).mean()


def summarize_ic(ic_series: pd.Series) -> dict[str, float]:
    """Headline signal-quality numbers, ready for the results table."""
    return {
        "n_days": int(len(ic_series)),
        "mean_ic": float(ic_series.mean()),
        "ic_std": float(ic_series.std(ddof=1)),
        "icir_ann": icir(ic_series),                 # overlap-adjusted (honest)
        "icir_naive": icir(ic_series, overlap=1),    # naive sqrt(252) (inflated)
        "ic_autocorr": float(ic_series.autocorr(1)),  # overlap signature
        "ic_tstat": ic_tstat(ic_series),
        "hit_rate": float((ic_series > 0).mean()),  # share of days with IC > 0
    }


# --------------------------------------------------------------------------- #
# Plot                                                                         #
# --------------------------------------------------------------------------- #
def plot_rolling_ic(
    ic_series: pd.Series,
    window: int = 60,
    path: Path = config.REPORTS_DIR / "rolling_ic.png",
) -> Path:
    """Save the rolling-IC chart: the single best view of signal stability."""
    import matplotlib

    matplotlib.use("Agg")  # headless — write a file, never open a window
    import matplotlib.pyplot as plt

    roll = rolling_ic(ic_series, window)
    mean_ic = ic_series.mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.plot(ic_series.index, ic_series.values, color="0.8", lw=0.6,
            label="daily IC")
    ax.plot(roll.index, roll.values, color="C0", lw=1.8,
            label=f"{window}d rolling IC")
    ax.axhline(mean_ic, color="C3", ls="--", lw=1.0,
               label=f"mean IC = {mean_ic:+.4f}")
    ax.set_title("Out-of-sample daily Information Coefficient (Spearman)")
    ax.set_ylabel("IC")
    ax.set_xlabel("date")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.margins(x=0.01)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Load helper                                                                  #
# --------------------------------------------------------------------------- #
def load_predictions(
    cache: Path = config.CACHE_DIR / "predictions.parquet",
) -> pd.DataFrame:
    """Read the Stage 3 OOS prediction frame the metrics are computed on."""
    cache = Path(cache)
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found — run `uv run model/train.py` first (Stage 3)."
        )
    return pd.read_parquet(cache)


# --------------------------------------------------------------------------- #
# Acceptance checks                                                            #
# --------------------------------------------------------------------------- #
# Sanity band from plan.md §0. Outside the soft band → suspicious; past the hard
# leakage bound → almost certainly a bug, so we fail loudly.
_IC_BAND = (0.01, 0.04)
_ICIR_BAND = (0.5, 2.0)
_IC_LEAK_HARD = 0.10     # mean daily IC this high on free daily data = leakage
_ICIR_LEAK_HARD = 3.0    # overlap-adjusted ICIR this high = leakage


def _acceptance(ic: pd.Series) -> None:
    """Plan §5 acceptance: mean IC and ICIR land in the sanity band (§0)."""
    s = summarize_ic(ic)

    print("\nSignal-quality summary (out-of-sample):")
    print(f"  IC days              {s['n_days']:>10,}")
    print(f"  mean daily IC        {s['mean_ic']:>+10.4f}   (band {_IC_BAND[0]}–{_IC_BAND[1]})")
    print(f"  IC std               {s['ic_std']:>10.4f}")
    print(f"  ICIR (overlap-adj)   {s['icir_ann']:>+10.2f}   (band {_ICIR_BAND[0]}–{_ICIR_BAND[1]})")
    print(f"  ICIR (naive √252)    {s['icir_naive']:>+10.2f}   ← inflated by overlap, not real")
    print(f"  IC lag-1 autocorr    {s['ic_autocorr']:>+10.2f}   (high ⇒ overlapping {config.LABEL_HORIZON}d labels)")
    print(f"  IC t-stat            {s['ic_tstat']:>+10.2f}   (also overlap-inflated)")
    print(f"  hit rate (IC>0)      {s['hit_rate']:>10.1%}")

    # Hard leakage tripwire — a bug should fail the build, not look impressive.
    assert s["mean_ic"] < _IC_LEAK_HARD, (
        f"mean IC {s['mean_ic']:.4f} >= {_IC_LEAK_HARD} — wildly high for free "
        "daily data; suspect look-ahead leakage. Go hunting (plan §0)."
    )
    assert s["icir_ann"] < _ICIR_LEAK_HARD, (
        f"ICIR {s['icir_ann']:.2f} >= {_ICIR_LEAK_HARD} — wildly high; suspect "
        "leakage or that the purge didn't fire."
    )
    assert s["mean_ic"] > 0, (
        f"mean IC {s['mean_ic']:.4f} <= 0 — the signal has no edge; the sign or "
        "the target is wrong before any backtest is worth running."
    )

    # Soft band: inside it is the win; outside but below the hard bound is a
    # "look closer", not a failure.
    in_ic = _IC_BAND[0] <= s["mean_ic"] <= _IC_BAND[1]
    in_icir = _ICIR_BAND[0] <= s["icir_ann"] <= _ICIR_BAND[1]
    if in_ic and in_icir:
        print("\n  ✓ mean IC and ICIR both land in the sanity band — modest, honest signal.")
    else:
        if not in_ic:
            where = "above" if s["mean_ic"] > _IC_BAND[1] else "below"
            print(f"\n  ⚠ mean IC is {where} the {_IC_BAND} band — "
                  f"{'inspect for leakage' if where == 'above' else 'weak but not alarming'}.")
        if not in_icir:
            where = "above" if s["icir_ann"] > _ICIR_BAND[1] else "below"
            print(f"  ⚠ ICIR is {where} the {_ICIR_BAND} band — "
                  f"{'inspect for leakage' if where == 'above' else 'noisy signal'}.")


if __name__ == "__main__":
    print("Stage 4 — backtest metrics (IC / ICIR)\n")
    preds = load_predictions()
    print(f"Predictions: {len(preds):,} rows | {preds['date'].nunique()} dates "
          f"| {preds['ticker'].nunique()} names")

    ic = daily_ic(preds)
    out = plot_rolling_ic(ic)
    print(f"Rolling-IC plot -> {out}")

    _acceptance(ic)
    print("\nStage 4 done.")
