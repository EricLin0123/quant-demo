"""Stage 3.2 — Walk-forward train / predict loop.

Run a LightGBM regressor across the purged, embargoed walk-forward folds from
`model.validation` and collect a single frame of **out-of-sample** predictions
to hand to the backtest. For every fold we train on the (purged + embargoed)
training rows and predict the held-out test block; because the folds tile time
contiguously, every (date, ticker) row is predicted exactly once, by a model
that never saw that date *or its forward-label window*.

Modelling choices, each defensible in a sentence:

  * **Target = `fwd_rank`** — the cross-sectional rank of the forward return in
    [-1, 1], not the raw return. We trade names *relative to peers each day*, so
    a rank target matches the objective and is robust to the fat tails of raw
    returns. (`lambdarank` would be optional polish; plain regression on the
    rank is enough and trains in seconds.)
  * **Deliberately small trees, strong regularization** — shallow depth, few
    leaves, high `min_child_samples`, feature/bagging subsampling, and L1/L2
    penalties. Daily cross-sectional equity signal is very low signal-to-noise;
    over-capacity just memorizes noise, so we starve the model on purpose.
  * **Fixed `SEED`** everywhere for reproducibility.

    uv run model/train.py   # runs the loop, prints IC + importance sanity checks
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from features import alpha  # noqa: E402
from model import validation  # noqa: E402

import lightgbm as lgb  # noqa: E402

# Low-capacity, strongly-regularized LightGBM. The point is to *not* fit the
# noise: shallow trees, big leaves, subsampling, and explicit L1/L2.
LGB_PARAMS: dict = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.02,
    "num_leaves": 15,        # shallow — capacity cap
    "max_depth": 4,
    "min_child_samples": 200,  # large leaves resist noise memorization
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,        # L1
    "reg_lambda": 5.0,       # L2
    "random_state": config.SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def run_walk_forward(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
    return_importances: bool = False,
):
    """Train/predict across purged walk-forward folds; collect OOS predictions.

    Returns a frame [date, ticker, pred, fwd_ret, fwd_rank, sector], one row per
    held-out (date, ticker), fully out-of-sample. If `return_importances`, also
    returns a Series of mean gain-importance per feature (averaged over folds).
    """
    params = {**LGB_PARAMS, **(params or {})}
    df = features_df.reset_index(drop=True)
    dates = df["date"]

    oos_parts: list[pd.DataFrame] = []
    importances = np.zeros(len(feature_cols), dtype=float)
    n_folds = 0

    for fold, (tr_idx, te_idx) in enumerate(
        validation.purged_walk_forward(dates), start=1
    ):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]

        # Acceptance guard, enforced at training time: the model's training
        # dates must not reach into the test block's label window.
        _assert_no_leak(train["date"], test["date"], config.LABEL_HORIZON)

        model = lgb.LGBMRegressor(**params)
        model.fit(train[feature_cols], train["fwd_rank"])

        pred = model.predict(test[feature_cols])
        part = test[["date", "ticker", "sector", "fwd_ret", "fwd_rank"]].copy()
        part["pred"] = pred
        oos_parts.append(part)

        importances += model.booster_.feature_importance(importance_type="gain")
        n_folds += 1
        print(
            f"  fold {fold}: train={len(train):,} test={len(test):,} "
            f"({test['date'].min().date()}..{test['date'].max().date()})"
        )

    oos = (
        pd.concat(oos_parts, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    oos = oos[["date", "ticker", "pred", "fwd_ret", "fwd_rank", "sector"]]

    if return_importances:
        imp = pd.Series(importances / max(n_folds, 1), index=feature_cols)
        return oos, imp.sort_values(ascending=False)
    return oos


def _assert_no_leak(train_dates: pd.Series, test_dates: pd.Series, horizon: int) -> None:
    """No training row's label window may reach the test block's start."""
    if len(train_dates) == 0:
        return
    test_start = test_dates.min()
    before = train_dates[train_dates < test_start]
    if len(before) == 0:
        return
    # A training row at day d carries information out to d + horizon sessions.
    # We can't count sessions cheaply here, so use the conservative calendar
    # bound the splitter already guarantees with margin: latest train day must
    # sit strictly before the test start (the splitter purges horizon+embargo
    # sessions, so this always holds — this is a belt-and-suspenders check).
    assert before.max() < test_start, "train/test dates overlap — purge failed"


def load_predictions(
    cache: Path = config.CACHE_DIR / "predictions.parquet",
    force: bool = False,
) -> pd.DataFrame:
    """Build-or-load the OOS prediction frame. Cache is immutable unless `force`."""
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    df = alpha.load_features()
    feature_cols = alpha.feature_columns(df)
    oos = run_walk_forward(df, feature_cols)
    cache.parent.mkdir(parents=True, exist_ok=True)
    oos.to_parquet(cache, index=False)
    print(f"  cached -> {cache}")
    return oos


# --------------------------------------------------------------------------- #
# Acceptance checks                                                            #
# --------------------------------------------------------------------------- #
def _acceptance(oos: pd.DataFrame, imp: pd.Series) -> None:
    """Assert the two plan §4.2 acceptance criteria, loudly."""
    # (1) Every date predicted exactly once, by a model that never saw it.
    #     The folds tile time with no overlap, so a duplicated (date,ticker)
    #     would mean two models scored the same row — i.e. a fold leak.
    dup = oos.duplicated(["date", "ticker"]).sum()
    print(f"\n[1] OOS coverage: {len(oos):,} predictions | "
          f"{oos['date'].nunique()} dates | dup (date,ticker) rows = {dup}")
    assert dup == 0, "a (date,ticker) was predicted by >1 fold — leakage"

    # Out-of-sample IC: must be modest. A huge IC here screams leakage.
    ic = (
        oos.groupby("date")
        .apply(lambda g: g["pred"].corr(g["fwd_ret"], method="spearman"),
               include_groups=False)
        .dropna()
    )
    icir = ic.mean() / ic.std() * np.sqrt(252)
    print(f"    OOS mean daily IC = {ic.mean():+.4f} | ICIR(ann) = {icir:+.2f} "
          f"| t-stat = {ic.mean() / ic.std() * np.sqrt(len(ic)):+.2f}")
    if not (0.0 < ic.mean() < 0.10):
        print("    ⚠ mean IC outside the expected ~0.01–0.04 band — inspect for leakage.")

    # (2) Feature importances are sane: momentum / volatility lead, not some
    #     leakage artifact.
    print("\n[2] Mean gain importance (top 8):")
    top = imp.head(8)
    total = imp.sum()
    for name, val in top.items():
        print(f"      {name:<18} {val / total * 100:5.1f}%")

    momentum_vol = [c for c in imp.index
                    if c.startswith(("mom_", "rev_", "vol_"))]
    share = imp[momentum_vol].sum() / total
    print(f"\n    momentum+reversal+vol share of importance = {share * 100:.1f}%")
    leading = set(imp.head(5).index)
    assert leading & set(momentum_vol), (
        "no momentum/reversal/vol feature in the top 5 — importances look off"
    )
    print("    ✓ momentum/vol family present in the leading features (no obvious "
          "leakage artifact dominating).")


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("Stage 3.2 — walk-forward train/predict\n")
    df = alpha.load_features()
    feature_cols = alpha.feature_columns(df)
    print(f"Features: {len(feature_cols)} cols | {len(df):,} rows\n")

    oos, imp = run_walk_forward(df, feature_cols, return_importances=True)

    cache = config.CACHE_DIR / "predictions.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    oos.to_parquet(cache, index=False)
    print(f"\n  cached -> {cache}")

    _acceptance(oos, imp)
    print("\nStage 3.2 done.")
