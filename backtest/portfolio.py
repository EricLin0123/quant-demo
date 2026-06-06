"""Stage 5 — Portfolio construction.

Turn the Stage 3 out-of-sample predictions into a tradeable weight schedule:
a cross-sectional, dollar-neutral, equal-weight quintile long-short book,
rebalanced every `REBALANCE_FREQ` sessions.

Design choices, each defensible in a sentence:

  * **Quintiles, not deciles.** 50 names / 5 = ~10 per leg. Deciles would leave
    ~5 names a side — too noisy to mean anything. (plan.md §0)
  * **Sector-neutral by construction.** With `SECTOR_NEUTRAL`, we *demean the
    prediction within its sector each day* before ranking. TSMC and the chip
    cluster dominate TWSE market cap, so a raw cross-sectional rank is half a
    disguised "long semiconductors" bet; demeaning strips the sector tilt so
    what's left is the within-sector view the model actually has an edge on.
  * **Dollar-neutral, equal-weight.** The long leg sums to +0.5 and the short
    leg to −0.5, so the book is exactly dollar-neutral (net 0) with 1.0 gross.
    Equal-weight inside each leg makes the fewest assumptions; with so few names
    a rank-weight scheme just concentrates risk in the noisiest tail name.
  * **Per-name cap** (`MAX_WEIGHT`) as a concentration guardrail. With 10
    equal-weight names a leg it doesn't bind (0.05 < 0.10); it's a safety rail
    that *would* bite if a leg ever shrank.

> **Short-availability caveat (say it out loud):** borrowing some TWSE names is
> hard/expensive in practice, so the long-short result is an *idealization*. The
> `long_only=True` top-quintile variant is the more realistic deployment; we
> build both so the deck can show the honest, shortable version too.

We emit weights **only on rebalance dates** ([date, ticker, weight]); the engine
(`backtest/engine.py`) holds them between rebalances. That keeps turnover — and
therefore cost — unambiguous: it's the change in *target*, nothing else.

    uv run backtest/portfolio.py   # builds weights + prints dollar-neutral checks
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `uv run backtest/portfolio.py` to import the top-level config.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def rebalance_dates(
    dates: pd.Series | pd.DatetimeIndex,
    freq: int = config.REBALANCE_FREQ,
) -> np.ndarray:
    """Every `freq`-th unique trading session, in order — when we re-strike weights."""
    unique = np.sort(pd.unique(pd.to_datetime(dates)))
    return unique[::freq]


def _leg_weights(score: pd.Series, n_quantiles: int, max_weight: float,
                 long_only: bool) -> pd.Series:
    """Equal-weight quintile weights for ONE date's cross-section of scores.

    Long top quintile (+), short bottom quintile (−). Long leg sums to +0.5 and
    short leg to −0.5 → dollar-neutral, 1.0 gross. `long_only` instead holds the
    top quintile fully invested (+1.0). Weights are capped at ±`max_weight` and
    each leg renormalized so the exposure target is preserved exactly.
    """
    n = len(score)
    k = max(n // n_quantiles, 1)                 # names per leg (~10 of 50)
    order = score.rank(method="first")           # break ties deterministically
    longs = order > (n - k)                      # top k
    shorts = order <= k                          # bottom k

    w = pd.Series(0.0, index=score.index)
    if long_only:
        w[longs] = 1.0 / longs.sum()             # fully invested long
        return _cap(w, max_weight, target=+1.0)

    w[longs] = 0.5 / longs.sum()
    w[shorts] = -0.5 / shorts.sum()
    # Cap each leg independently so the long/short balance (dollar-neutrality)
    # is preserved even if the cap binds.
    w[longs] = _cap(w[longs], max_weight, target=+0.5)
    w[shorts] = _cap(w[shorts], max_weight, target=-0.5)
    return w


def _cap(w: pd.Series, max_weight: float, target: float) -> pd.Series:
    """Clip |weight| to `max_weight`, then rescale so the leg sums to `target`."""
    capped = w.clip(-max_weight, max_weight)
    s = capped.sum()
    return capped * (target / s) if s != 0 else capped


def build_weights(
    preds_df: pd.DataFrame,
    n_quantiles: int = config.N_QUANTILES,
    sector_neutral: bool = config.SECTOR_NEUTRAL,
    max_weight: float = config.MAX_WEIGHT,
    rebalance_freq: int = config.REBALANCE_FREQ,
    long_only: bool = False,
) -> pd.DataFrame:
    """Target weights on each rebalance date. Returns [date, ticker, weight].

    On every rebalance session: optionally sector-demean `pred`, rank the
    cross-section, and assign equal-weight quintile long-short (or long-only)
    weights. Only non-zero positions are returned; the engine fills the rest
    with 0 and forward-fills between rebalances.
    """
    df = preds_df[["date", "ticker", "pred", "sector"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    rebals = set(pd.Timestamp(d) for d in rebalance_dates(df["date"], rebalance_freq))
    df = df[df["date"].isin(rebals)]

    if sector_neutral:
        # Demean the prediction within (date, sector): strip the sector mean so
        # the residual ranks names *against their own sector peers*.
        df["score"] = df["pred"] - df.groupby(["date", "sector"])["pred"].transform("mean")
    else:
        df["score"] = df["pred"]

    parts: list[pd.DataFrame] = []
    for date, g in df.groupby("date"):
        w = _leg_weights(g.set_index("ticker")["score"], n_quantiles,
                         max_weight, long_only)
        w = w[w != 0.0]
        parts.append(pd.DataFrame({"date": date, "ticker": w.index, "weight": w.values}))

    weights = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])
    return weights.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Acceptance checks                                                            #
# --------------------------------------------------------------------------- #
def _acceptance(weights: pd.DataFrame, long_only: bool = False) -> None:
    """plan §6 acceptance: weights sum to ~0 each day (dollar-neutral)."""
    day_sum = weights.groupby("date")["weight"].sum()
    gross = weights.groupby("date")["weight"].apply(lambda w: w.abs().sum())
    leg = weights.groupby("date")["weight"].apply(
        lambda w: ((w > 0).sum(), (w < 0).sum()))

    target_net = 1.0 if long_only else 0.0
    # Smallest a leg ever gets → the inherent equal-weight concentration floor
    # (1/k). A fully-invested long-only leg of k<1/cap names *can't* fit under
    # the cap, so the feasible bound is max(cap, 1/k), not the cap alone.
    min_leg = min(min(l, s) if not long_only else l for l, s in leg)
    eff_cap = max(config.MAX_WEIGHT, 1.0 / min_leg)

    print(f"\n  rebalance dates           {weights['date'].nunique()}")
    print(f"  net exposure  (min..max)  {day_sum.min():+.2e} .. {day_sum.max():+.2e} "
          f"(target {target_net:+.0f})")
    print(f"  gross exposure(min..max)  {gross.min():.3f} .. {gross.max():.3f}")
    print(f"  names/leg (long, short)   {leg.iloc[0]} (first rebalance)")
    print(f"  max |name weight|         {weights['weight'].abs().max():.3f} "
          f"(cap {config.MAX_WEIGHT}, feasible {eff_cap:.3f})")

    assert np.allclose(day_sum.to_numpy(), target_net, atol=1e-9), (
        "weights are not dollar-neutral — net exposure drifts from target"
    )
    assert weights["weight"].abs().max() <= eff_cap + 1e-9, (
        "a name exceeds the feasible per-name weight cap"
    )
    note = "" if eff_cap <= config.MAX_WEIGHT else " (cap slack: too few names/leg to fully invest under it)"
    print(f"  ✓ dollar-neutral within 1e-9 and within the feasible per-name cap.{note}")


if __name__ == "__main__":
    from backtest import metrics  # noqa: E402 (reuse the prediction loader)

    print("Stage 5 — portfolio construction\n")
    preds = metrics.load_predictions()

    print("Long-short, sector-neutral:")
    w_ls = build_weights(preds)
    _acceptance(w_ls)

    print("\nLong-only top-quintile (realistic, shortable deployment):")
    w_lo = build_weights(preds, long_only=True)
    _acceptance(w_lo, long_only=True)

    print("\nStage 5 done.")
