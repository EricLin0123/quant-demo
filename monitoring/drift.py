"""Stage 7 — Monitoring / drift: the three-ring detection framework.

The same three layers I'd put on any production ML system, restated for a
delayed-label trading setting:

  1. **Performance ring** — rolling 60-day IC. Is the signal itself decaying?
     This is the windowed-performance layer. In markets the label is delayed
     (here by `LABEL_HORIZON` days), so performance always lags reality — which
     is *exactly* why it can't be the only ring.
  2. **Input-drift ring** — PSI + KS per feature, early period vs late period.
     Has the input distribution moved under the model (covariate shift)? This
     fires immediately on new data, with no label needed, so it's the early
     warning the performance ring can't be. We run it on the **raw** features:
     the production features are cross-sectionally z-scored each day, which
     deliberately standardizes away level/scale drift — so the raw frame is
     where covariate shift over the decade actually shows up.
  3. **Regime ring** — a realized-volatility percentile flag, so degraded
     stretches are *explainable* ("we were in the top-vol decile") rather than
     mysterious.

The batch statistical artifact is an Evidently data-drift report rendered to
`reports/drift.html` — a real file to flash on a slide.

> **Defense line:** "Real-time input-drift proxy, windowed performance, batch
> statistical report — three independent layers. The label being delayed is the
> whole reason you can't lean on performance alone and need the input ring."

    uv run monitoring/drift.py   # writes reports/monitoring.png + reports/drift.html
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `uv run monitoring/drift.py` to import the top-level config + siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# PSI rule-of-thumb bands (industry standard): < 0.1 stable, 0.1–0.25 moderate
# shift, > 0.25 significant shift.
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25


# --------------------------------------------------------------------------- #
# Ring 1 — performance (rolling IC)                                            #
# --------------------------------------------------------------------------- #
def rolling_ic(ic_series: pd.Series, window: int = 60) -> pd.Series:
    """Trailing-`window` mean IC — the windowed-performance ring."""
    return ic_series.rolling(window, min_periods=window // 2).mean()


def ic_decay(ic_series: pd.Series) -> dict:
    """Quantify signal decay: first-half vs second-half mean IC and OLS slope."""
    half = len(ic_series) // 2
    early, late = ic_series.iloc[:half], ic_series.iloc[half:]
    x = np.arange(len(ic_series), dtype=float)
    slope = float(np.polyfit(x, ic_series.to_numpy(), 1)[0])
    return {
        "ic_first_half": float(early.mean()),
        "ic_second_half": float(late.mean()),
        "ic_slope_per_day": slope,           # change in daily IC per session
        "ic_slope_per_year": slope * 252,
    }


# --------------------------------------------------------------------------- #
# Ring 2 — input drift (PSI / KS)                                              #
# --------------------------------------------------------------------------- #
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two samples of one feature.

    Bin edges come from quantiles of the `expected` (reference) sample so each
    reference bin holds ~equal mass; PSI = Σ (aᵢ − eᵢ)·ln(aᵢ / eᵢ) over bins.
    """
    expected = expected.dropna().to_numpy()
    actual = actual.dropna().to_numpy()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf       # catch tails outside ref range
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)

    eps = 1e-6                                  # avoid log(0) / divide-by-zero
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(
    df_early: pd.DataFrame,
    df_late: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    """Per-feature PSI + two-sample KS, early vs late. Sorted worst-drift first."""
    from scipy import stats

    rows = []
    for col in feature_cols:
        ks = stats.ks_2samp(df_early[col].dropna(), df_late[col].dropna())
        p = psi(df_early[col], df_late[col], bins)
        band = ("significant" if p >= PSI_SIGNIFICANT
                else "moderate" if p >= PSI_MODERATE else "stable")
        rows.append({"feature": col, "psi": p, "ks_stat": float(ks.statistic),
                     "ks_pvalue": float(ks.pvalue), "drift": band})
    return (pd.DataFrame(rows)
            .sort_values("psi", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Ring 3 — regime (realized-vol percentile)                                    #
# --------------------------------------------------------------------------- #
def regime_flag(
    returns: pd.DataFrame,
    window: int = 21,
    pct: float = 0.80,
) -> pd.DataFrame:
    """Flag high-volatility regimes from the equal-weight universe return.

    Returns [date, realized_vol, vol_pct, high_vol]; `high_vol` marks days whose
    trailing annualized vol sits above the `pct` percentile of the sample — the
    stretches where any signal is most likely to wobble.
    """
    mkt = returns.mean(axis=1)                       # equal-weight market proxy
    rv = mkt.rolling(window).std() * np.sqrt(252)
    rv = rv.dropna()
    thr = rv.quantile(pct)
    return pd.DataFrame({
        "date": rv.index,
        "realized_vol": rv.to_numpy(),
        "vol_pct": rv.rank(pct=True).to_numpy(),
        "high_vol": (rv > thr).to_numpy(),
    })


# --------------------------------------------------------------------------- #
# Batch artifact — Evidently data-drift report                                 #
# --------------------------------------------------------------------------- #
def evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    path: Path = config.REPORTS_DIR / "drift.html",
    max_rows: int = 15000,
) -> Path:
    """Render an Evidently data-drift report (reference vs current) to HTML."""
    warnings.filterwarnings("ignore")
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    # Cap rows so the HTML stays light and the run stays fast; seeded for repro.
    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > max_rows:
            return df.sample(max_rows, random_state=config.SEED)
        return df

    ref = _sample(reference_df[feature_cols])
    cur = _sample(current_df[feature_cols])

    data_def = DataDefinition(numerical_columns=list(feature_cols))
    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)

    snapshot = Report([DataDriftPreset()]).run(current_data=cur_ds,
                                               reference_data=ref_ds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return path


# --------------------------------------------------------------------------- #
# Orchestration + plot                                                         #
# --------------------------------------------------------------------------- #
def split_early_late(df: pd.DataFrame, frac: float = 0.33) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a date-sorted frame into the earliest and latest `frac` by date."""
    days = np.sort(df["date"].unique())
    lo = days[int(len(days) * frac) - 1]
    hi = days[int(len(days) * (1 - frac))]
    return df[df["date"] <= lo].copy(), df[df["date"] >= hi].copy()


def plot_monitoring(
    ic: pd.Series,
    drift: pd.DataFrame,
    regime: pd.DataFrame,
    window: int = 60,
    path: Path = config.REPORTS_DIR / "monitoring.png",
) -> Path:
    """One slide, three rings: rolling IC (+ regime shading), PSI bars, vol."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10))
    hi = regime[regime["high_vol"]]["date"]

    # Ring 1 — rolling IC, with high-vol stretches shaded.
    roll = rolling_ic(ic, window)
    ax1.axhline(0.0, color="0.6", lw=0.8)
    ax1.plot(roll.index, roll.values, color="C0", lw=1.6, label=f"{window}d rolling IC")
    ax1.axhline(ic.mean(), color="C3", ls="--", lw=1.0, label=f"mean {ic.mean():+.3f}")
    for d in hi:
        ax1.axvspan(d, d, color="0.85", lw=0)
    ax1.set_title("Ring 1 · performance — rolling IC (grey = high-vol regime)")
    ax1.set_ylabel("IC"); ax1.legend(loc="upper left", fontsize=8); ax1.margins(x=0.01)

    # Ring 2 — top input-drift features by PSI.
    top = drift.head(10).iloc[::-1]
    colors = ["C3" if p >= PSI_SIGNIFICANT else "C1" if p >= PSI_MODERATE else "C0"
              for p in top["psi"]]
    ax2.barh(top["feature"], top["psi"], color=colors)
    ax2.axvline(PSI_MODERATE, color="C1", ls="--", lw=0.8, label="moderate 0.10")
    ax2.axvline(PSI_SIGNIFICANT, color="C3", ls="--", lw=0.8, label="significant 0.25")
    ax2.set_title("Ring 2 · input drift — PSI per raw feature (early vs late)")
    ax2.set_xlabel("PSI"); ax2.legend(loc="lower right", fontsize=8)

    # Ring 3 — realized vol with high-vol threshold.
    ax3.plot(regime["date"], regime["realized_vol"], color="C0", lw=1.0)
    thr = regime.loc[~regime["high_vol"], "realized_vol"].max()
    ax3.axhline(thr, color="C3", ls="--", lw=0.9, label="high-vol threshold")
    ax3.fill_between(regime["date"], regime["realized_vol"], thr,
                     where=regime["high_vol"], color="C3", alpha=0.25)
    ax3.set_title("Ring 3 · regime — annualized realized vol (equal-weight universe)")
    ax3.set_ylabel("ann. vol"); ax3.set_xlabel("date")
    ax3.legend(loc="upper left", fontsize=8); ax3.margins(x=0.01)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def run_monitoring(
    predictions: pd.DataFrame | None = None,
    raw_features: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    make_html: bool = True,
) -> dict:
    """Run all three rings + the Evidently artifact; return a summary dict."""
    from backtest import metrics
    from backtest import engine
    from features import alpha

    if predictions is None:
        predictions = metrics.load_predictions()
    if raw_features is None:
        raw_features = alpha.build_raw_features()
    if returns is None:
        returns = engine.daily_returns()

    feature_cols = alpha.feature_columns(raw_features)

    # Ring 1
    ic = metrics.daily_ic(predictions)
    decay = ic_decay(ic)

    # Ring 2
    early, late = split_early_late(raw_features)
    drift = feature_psi(early, late, feature_cols)

    # Ring 3
    regime = regime_flag(returns)

    plot_path = plot_monitoring(ic, drift, regime)
    html_path = (evidently_report(early, late, feature_cols) if make_html else None)

    return {
        "ic_decay": decay,
        "drift": drift,
        "n_drifted": int((drift["psi"] >= PSI_MODERATE).sum()),
        "regime": regime,
        "high_vol_frac": float(regime["high_vol"].mean()),
        "monitoring_png": plot_path,
        "drift_html": html_path,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Stage 7 — monitoring / drift (three-ring framework)\n")
    res = run_monitoring()

    d = res["ic_decay"]
    print("Ring 1 · performance (rolling IC):")
    print(f"  mean IC  first half / second half : "
          f"{d['ic_first_half']:+.4f} / {d['ic_second_half']:+.4f}")
    print(f"  IC trend (per year)               : {d['ic_slope_per_year']:+.4f}")

    print("\nRing 2 · input drift (early vs late, raw features):")
    print(res["drift"].to_string(index=False,
          formatters={"psi": "{:.3f}".format, "ks_stat": "{:.3f}".format,
                      "ks_pvalue": "{:.1e}".format}))
    print(f"  features with PSI ≥ {PSI_MODERATE}: {res['n_drifted']} / {len(res['drift'])}")

    print(f"\nRing 3 · regime: high-vol days = {res['high_vol_frac']:.1%} of sample")

    print(f"\nArtifacts:\n  {res['monitoring_png']}\n  {res['drift_html']}")
    assert Path(res["monitoring_png"]).exists(), "monitoring plot not written"
    assert res["drift_html"] and Path(res["drift_html"]).exists(), "drift html not written"
    print("\nStage 7 done.")
