"""Diagnostic plots. All saved to reports/ with a non-interactive backend.

Per §10/§12 the reported metrics are strictly per-next-day; the compounded price path
here is a **sanity-only** visual gut-check, explicitly excluded from scoring.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def candles(df_ohlcv: pd.DataFrame, fname: str = "test_candles.png") -> str:
    import mplfinance as mpf

    p = config.REPORTS / fname
    mpf.plot(
        df_ohlcv[["Open", "High", "Low", "Close", "Volume"]],
        type="candle", volume=True, style="yahoo",
        title="TWII — test window", savefig=dict(fname=str(p), dpi=130),
    )
    logger.info("saved %s", p)
    return str(p)


def pred_vs_actual(y_true, y_pred, ic: float, fname: str = "pred_vs_actual.png") -> str:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4)
    lim = max(np.abs(y_true).max(), np.abs(y_pred).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y = x")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("actual next-day log return")
    ax.set_ylabel("predicted next-day log return")
    ax.set_title(f"Predicted vs. actual (IC = {ic:+.4f})")
    ax.legend()
    p = config.REPORTS / fname
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    logger.info("saved %s", p)
    return str(p)


def residual_hist(y_true, y_pred, fname: str = "residuals.png") -> str:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(resid, bins=60, alpha=0.8)
    ax.set_title(f"Residuals (std={resid.std():.4e}, kurtosis check by eye)")
    ax.set_xlabel("actual - predicted")
    p = config.REPORTS / fname
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    logger.info("saved %s", p)
    return str(p)


def confusion_matrix(cm, metrics: dict, fname: str = "direction_confusion.png") -> str:
    """Heatmap of the binary up/down directional confusion matrix.

    `cm` is the DataFrame from `evaluate.direction_confusion` (rows actual, cols pred).
    Cells are annotated with counts; the title carries the directional accuracy.
    """
    import pandas as pd

    cm = pd.DataFrame(cm)
    values = cm.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(values, cmap="Blues")

    ax.set_xticks(range(cm.shape[1]), labels=list(cm.columns))
    ax.set_yticks(range(cm.shape[0]), labels=list(cm.index))
    ax.set_xlabel("predicted direction")
    ax.set_ylabel("actual direction")

    vmax = values.max() if values.size else 1.0
    total = values.sum() if values.size else 1.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            n = int(values[i, j])
            pct = 100.0 * n / total if total else 0.0
            ax.text(
                j, i, f"{n}\n{pct:.1f}%", ha="center", va="center",
                color="white" if values[i, j] > 0.6 * vmax else "black",
                fontsize=11,
            )
    acc = metrics.get("accuracy", float("nan"))
    ax.set_title(
        f"Directional confusion (acc={acc:+.4f}, n={metrics.get('n','?')})\n"
        f"pred up rate={metrics.get('pred_up_rate', float('nan')):.3f}, "
        f"actual up rate={metrics.get('actual_up_rate', float('nan')):.3f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = config.REPORTS / fname
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    logger.info("saved %s", p)
    return str(p)


def actual_vs_predicted_timeseries(
    dates, y_true, y_pred, fname: str = "actual_vs_predicted_timeseries.png"
) -> str:
    """Per-test-day directional scorecard: was the model right or wrong each day?

    The model's *predicted magnitude* is near-constant and not meaningful, so we drop
    it entirely and visualise only what matters: the **direction call**. Each actual
    next-day return is a bar coloured **green when the model got the sign right** and
    **red when it missed**. A running cumulative hit-rate line is overlaid against a
    0.5 coin-flip reference — the whole point is that it never pulls away from chance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    hit = np.sign(y_pred) == np.sign(y_true)
    dir_acc = float(np.mean(hit)) if len(hit) else float("nan")
    running = np.cumsum(hit) / np.arange(1, len(hit) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 4.8))

    # actual returns as bars, coloured by whether the model called the direction right
    colors = np.where(hit, "tab:green", "tab:red")
    ax1.bar(dates, y_true, width=1.0, color=colors)
    ax1.axhline(0, color="0.4", lw=0.6)
    ax1.set_ylabel("actual next-day log return")
    ax1.margins(x=0.01)

    # running directional hit rate vs the coin-flip line, on a twin axis
    ax2 = ax1.twinx()
    ax2.plot(dates, running, color="black", lw=1.4, label="cumulative hit rate")
    ax2.axhline(0.5, color="black", ls="--", lw=1.0, alpha=0.7, label="coin flip (0.50)")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("cumulative directional hit rate")

    # legend: explain the bar colours + the running line
    from matplotlib.patches import Patch
    handles = [
        Patch(color="tab:green", label="direction hit"),
        Patch(color="tab:red", label="direction miss"),
        *ax2.get_legend_handles_labels()[0],
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=8, ncol=4)

    ax1.set_title(
        f"Was the next-day direction right? Green = hit, red = miss "
        f"(overall hit rate = {dir_acc:.1%} — barely a coin flip)"
    )

    p = config.REPORTS / fname
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    logger.info("saved %s", p)
    return str(p)


def temporal_consistency(
    auc_table: pd.DataFrame, tau: float | None = None,
    fname: str = "temporal_consistency.png",
) -> str:
    """Horizontal bar chart of each feature's temporal-consistency AUC.

    Replaces the raw keep/drop grid with a colour-coded visual: one bar per feature,
    length = aggregated AUC, coloured on a stable→drift scale (green ≈ 0.5 stable,
    red → 1.0 drift). A dashed line marks the keep/drop threshold τ; dropped features
    are flagged so the eye lands on the drift immediately. Rows are most-stable first.
    """
    tau = config.TCA_TAU if tau is None else tau
    t = pd.DataFrame(auc_table).sort_values("agg_auc").reset_index(drop=True)
    feats = t["feature"].astype(str).tolist()
    aucs = t["agg_auc"].to_numpy(dtype=float)
    keep = t["keep"].to_numpy(dtype=bool)

    # colour each bar by its AUC on a stable(green)→drift(red) ramp
    cmap = plt.get_cmap("RdYlGn_r")
    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.0)
    colors = cmap(norm(np.clip(aucs, 0.5, 1.0)))

    y = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.34 * len(feats) + 1.2)))
    ax.barh(y, aucs, color=colors, edgecolor="0.3", linewidth=0.4)
    ax.set_yticks(y, labels=feats)
    ax.invert_yaxis()  # most-stable on top
    ax.set_xlim(0.5, max(1.0, float(aucs.max()) + 0.02))
    ax.set_xlabel("aggregated temporal-consistency AUC  (0.5 = stable → 1.0 = drift)")

    ax.axvline(tau, color="black", ls="--", lw=1.2)
    ax.text(tau + 0.005, len(feats) - 0.5, f"keep ↤ τ = {tau:g} ↦ drop",
            ha="left", va="bottom", fontsize=9, color="0.25")

    # annotate each bar with its value; mark dropped features
    for yi, a, k in zip(y, aucs, keep):
        ax.text(a + 0.005, yi, f"{a:.3f}" + ("" if k else "  ✗ drop"),
                va="center", ha="left", fontsize=8,
                color="0.2" if k else "tab:red")

    n_drop = int((~keep).sum())
    ax.set_title(
        f"Temporal-consistency feature selection — {len(feats)} features, "
        f"{n_drop} dropped (AUC > {tau:g})"
    )
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.046, pad=0.02, label="drift")
    p = config.REPORTS / fname
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    logger.info("saved %s", p)
    return str(p)


def compounded_sanity(
    dates, y_true, y_pred, fname: str = "compounded_sanity.png"
) -> str:
    """SANITY-ONLY: compounded price path from predicted vs actual returns.

    Explicitly NOT a reported metric (see §10). Drawn purely as a visual gut-check.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    actual_path = np.exp(np.cumsum(y_true))
    pred_path = np.exp(np.cumsum(y_pred))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, actual_path, label="actual (compounded)")
    ax.plot(dates, pred_path, label="predicted (compounded)", alpha=0.8)
    ax.set_title("SANITY ONLY — compounded path (excluded from metrics)")
    ax.set_ylabel("growth of 1 (test start)")
    ax.legend()
    p = config.REPORTS / fname
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    logger.info("saved %s", p)
    return str(p)
