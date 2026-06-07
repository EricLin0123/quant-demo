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
    """Per-test-day actual vs predicted next-day return over time.

    Because predicted magnitudes are ~30x smaller than actuals (the model shrinks
    toward the conditional median), the two are drawn on **separate y-axes** so the
    predicted *shape/timing* is visible alongside the actual series. Predicted markers
    are coloured by directional hit (green = sign matched the actual move, red = missed),
    which is the per-day view of the confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    hit = np.sign(y_pred) == np.sign(y_true)

    fig, ax1 = plt.subplots(figsize=(12, 4.8))

    # actual on the primary axis (grey bars)
    ax1.bar(dates, y_true, width=1.0, color="0.75", label="actual")
    ax1.axhline(0, color="0.4", lw=0.6)
    ax1.set_ylabel("actual next-day log return", color="0.4")
    ax1.tick_params(axis="y", labelcolor="0.4")

    # predicted on a twin axis (line + hit/miss markers)
    ax2 = ax1.twinx()
    ax2.plot(dates, y_pred, color="tab:blue", lw=0.9, alpha=0.7, label="predicted")
    ax2.scatter(np.asarray(dates)[hit], y_pred[hit], s=14,
                color="tab:green", label="direction hit", zorder=3)
    ax2.scatter(np.asarray(dates)[~hit], y_pred[~hit], s=14,
                color="tab:red", label="direction miss", zorder=3)
    ax2.set_ylabel("predicted next-day log return", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    dir_acc = float(np.mean(hit)) if len(hit) else float("nan")
    ax1.set_title(
        f"Test window: actual vs. predicted next-day return per day "
        f"(directional hit rate = {dir_acc:+.4f}; note the ~30x y-axis scale gap)"
    )
    # merged legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2)

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
