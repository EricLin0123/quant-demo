"""Stage 3 — Purged, embargoed walk-forward validation.

The single most important methodological control in this repo. The target is a
forward return over `LABEL_HORIZON` trading days, so the label of a sample at
day `t` is computed from prices out to `t + LABEL_HORIZON`. Two consecutive
samples therefore have *overlapping* label windows, and a naive split that puts
day `t` in train and day `t + 1` in test leaks future information backwards:
the training label already "saw" days that belong to the test window.

Two controls fix this (López de Prado, *Advances in Financial ML*, ch. 7):

  * **Purge** — drop training rows whose label window overlaps the test window.
    A training row at day `d` has label window [d, d + horizon]; it overlaps a
    test window starting at `test_start` whenever `d >= test_start - horizon`.
    So we purge the `horizon` trading days immediately before each test fold.
  * **Embargo** — additionally drop `embargo_days` of training rows on *either*
    side of the test window. This guards against subtler leakage (serially
    correlated features / slow-moving regimes) bleeding across the boundary.

Critically, purge/embargo are measured in **trading days** (positions in the
sorted calendar of unique session dates), not calendar days. The plan sketch
used `pd.Timedelta(days=...)`, but 10 trading days is ~14 calendar days, so a
calendar-based gap would under-purge across every weekend and holiday. We index
by trading session instead.

This is a *walk-forward* splitter: each successive fold tests a later, contiguous
block of time. Unlike plain expanding-window CV we keep usable training data on
*both* sides of the test block (minus the purge+embargo halo) so later folds
aren't starved — but training rows never overlap the test label window.

    uv run model/validation.py   # prints a fold table + leakage self-check
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# Allow `uv run model/validation.py` to import the top-level config.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def purged_walk_forward(
    dates: pd.Series | np.ndarray | pd.DatetimeIndex,
    n_splits: int = config.N_SPLITS,
    embargo_days: int = config.EMBARGO_DAYS,
    label_horizon: int = config.LABEL_HORIZON,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) integer-position arrays for each fold.

    `dates` is the per-row date column of the feature frame (one entry per
    (date, ticker) row, so dates repeat). The unique trading sessions are split
    into `n_splits + 1` contiguous chunks; chunks 1..n each serve as one test
    fold (the first chunk is training-only warmup), giving `n_splits` folds.

    Around each test fold we carve a halo out of the training set:
      * before the fold: purge `label_horizon` sessions (overlapping labels)
        plus `embargo_days` sessions of embargo;
      * after the fold:  purge `embargo_days` sessions of embargo.

    Indices are positions into the *original* `dates` array, so the caller can
    do `df.iloc[train_idx]` / `df.iloc[test_idx]` directly.
    """
    dates = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))
    date_vals = dates.to_numpy()

    unique_days = np.sort(np.unique(date_vals))
    n_days = len(unique_days)
    folds = np.array_split(unique_days, n_splits + 1)

    for k in range(1, len(folds)):
        test_days = folds[k]
        if len(test_days) == 0:
            continue
        test_start, test_end = test_days[0], test_days[-1]

        # Positions of the test block within the trading calendar.
        i_start = int(np.searchsorted(unique_days, test_start, side="left"))
        i_end = int(np.searchsorted(unique_days, test_end, side="right")) - 1

        # Lower purge boundary: keep training strictly before this session.
        lo_pos = i_start - (label_horizon + embargo_days)
        purge_lo = unique_days[lo_pos] if lo_pos >= 0 else None

        # Upper embargo boundary: keep training strictly after this session.
        hi_pos = i_end + embargo_days
        purge_hi = unique_days[hi_pos] if hi_pos < n_days else None

        before = (date_vals < purge_lo) if purge_lo is not None else np.zeros(len(date_vals), bool)
        after = (date_vals > purge_hi) if purge_hi is not None else np.zeros(len(date_vals), bool)
        train_mask = before | after
        test_mask = (date_vals >= test_start) & (date_vals <= test_end)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def naive_kfold(
    dates: pd.Series | np.ndarray | pd.DatetimeIndex,
    n_splits: int = config.N_SPLITS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The leaky baseline: same contiguous folds, but **no purge or embargo**.

    Kept so Stage 3 can run the "demo-the-failure" comparison — train on every
    row that isn't in the test block (including the rows whose labels overlap
    it) and watch IC inflate. Presenting that inflated number as a cautionary
    tale, next to the purged number, is the point.
    """
    dates = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))
    date_vals = dates.to_numpy()
    unique_days = np.sort(np.unique(date_vals))
    folds = np.array_split(unique_days, n_splits + 1)

    for k in range(1, len(folds)):
        test_days = folds[k]
        if len(test_days) == 0:
            continue
        test_mask = (date_vals >= test_days[0]) & (date_vals <= test_days[-1])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


# --------------------------------------------------------------------------- #
# Self-check / demo                                                            #
# --------------------------------------------------------------------------- #
def _fold_report(dates: pd.Series) -> None:
    """Print a per-fold table and assert the no-leakage invariant."""
    dates = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))
    date_vals = dates.to_numpy()
    horizon = config.LABEL_HORIZON
    unique_days = np.sort(np.unique(date_vals))

    print(
        f"purged_walk_forward: {config.N_SPLITS} folds | "
        f"horizon={horizon}d | embargo={config.EMBARGO_DAYS}d | "
        f"{len(unique_days)} trading sessions "
        f"{pd.Timestamp(unique_days[0]).date()} -> {pd.Timestamp(unique_days[-1]).date()}\n"
    )
    header = f"{'fold':>4} {'train rows':>11} {'test rows':>10}  {'test window':>25}  {'gap to train (sessions)':>24}"
    print(header)
    print("-" * len(header))

    for i, (tr, te) in enumerate(purged_walk_forward(dates), start=1):
        tr_days = np.unique(date_vals[tr])
        te_days = np.unique(date_vals[te])
        te_start, te_end = te_days[0], te_days[-1]

        # No training row's label window may reach into the test window;
        # the exact check uses trading-session positions, not calendar days.
        pos = {d: j for j, d in enumerate(unique_days)}
        i_te_start = pos[te_start]
        before = tr_days[tr_days < te_start]
        gap_sessions = (
            i_te_start - max(pos[d] for d in before) if len(before) else float("nan")
        )

        # Invariant: every training day before the test fold ends its label
        # window strictly before the test fold begins (purged + embargoed).
        for d in before:
            assert pos[d] + horizon < i_te_start, (
                f"fold {i}: training day {pd.Timestamp(d).date()} label window "
                f"overlaps test start {pd.Timestamp(te_start).date()}"
            )

        win = f"{pd.Timestamp(te_start).date()}..{pd.Timestamp(te_end).date()}"
        print(f"{i:>4} {len(tr):>11,} {len(te):>10,}  {win:>25}  {gap_sessions:>24}")

    print("\nLeakage invariant holds: no training label window overlaps any test fold.")

    # Show the failure mode the controls prevent.
    naive_train = sum(len(tr) for tr, _ in naive_kfold(dates))
    purged_train = sum(len(tr) for tr, _ in purged_walk_forward(dates))
    print(
        f"\nDemo-the-failure: naive k-fold keeps {naive_train:,} train rows vs "
        f"{purged_train:,} after purge+embargo "
        f"({naive_train - purged_train:,} rows carry overlapping-label leakage)."
    )


if __name__ == "__main__":
    # Prefer the real feature frame; fall back to a synthetic calendar so the
    # splitter is testable before the Stage 2 cache exists.
    try:
        from features import alpha  # noqa: E402

        df = alpha.load_features()
        dates = df["date"]
        print("Using Stage 2 feature frame.\n")
    except Exception as e:  # pragma: no cover - convenience path
        print(f"(feature cache unavailable: {e}; using synthetic calendar)\n")
        cal = pd.bdate_range("2018-01-01", "2024-12-31")
        dates = pd.Series(np.repeat(cal.values, 50))  # 50 names per day

    _fold_report(dates)
    print("\nStage 3.1 (validation splitter) done.")
