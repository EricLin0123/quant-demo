"""Plots: candles, pred-vs-actual, residuals, and an illustrative price path.

All functions return a matplotlib Figure so the notebook controls rendering.
Each closes its figure with ``plt.close(fig)`` before returning, so the inline
backend does not also auto-render the pyplot-managed copy at cell end (which
would draw every plot twice).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from . import config


def candles(df: pd.DataFrame, test_index: pd.Index, use_volume: bool = True):
    """mplfinance candlestick of the test window."""
    window = df.loc[df.index.isin(test_index), ["Open", "High", "Low", "Close", "Volume"]]
    fig, _ = mpf.plot(
        window, type="candle", style="yahoo",
        volume=use_volume, returnfig=True,
        title=f"{config.TICKER} — test window",
        ylabel="Index Level", ylabel_lower="Volume",
        figsize=(11, 6),
    )
    plt.close(fig)
    return fig


def pred_vs_actual(y_true: pd.Series, y_pred: np.ndarray, ic: float):
    """Scatter of predicted vs. actual next-day returns, IC annotated."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5)
    lim = max(np.abs(y_true).max(), np.abs(y_pred).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y = x")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("actual next-day log return")
    ax.set_ylabel("predicted")
    ax.set_title(f"Predicted vs. actual  (IC = {ic:.3f})")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def residual_hist(y_true: pd.Series, y_pred: np.ndarray, bins: int = 60):
    """Residual histogram — eyeball the fat tails."""
    resid = y_true.to_numpy() - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(resid, bins=bins, alpha=0.8)
    ax.axvline(0, color="r", lw=1)
    ax.set_xlabel("residual (actual − predicted)")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution")
    fig.tight_layout()
    plt.close(fig)
    return fig


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray, window: int = 21):
    """Single-period (next-day) directional accuracy — NOT a compounded level.

    Each test day is an independent up/down call: a hit is
    ``sign(pred) == sign(actual)``. We deliberately avoid compounding predicted
    returns into a price path, because a small per-step return bias compounds
    into a large, misleading level divergence over the test window. Direction is
    evaluated one day at a time and never accumulated.

    Left panel: rolling hit-rate vs. the 50% coin-flip line, with the overall
    rate annotated. Right panel: the up/down confusion matrix.
    """
    yt = y_true.to_numpy()
    yp = np.asarray(y_pred)

    # Exclude exact-zero predictions (no directional call was made).
    called = yp != 0
    hit = (np.sign(yp) == np.sign(yt)) & called
    overall = float(hit[called].mean()) if called.any() else float("nan")

    hit_series = pd.Series(np.where(called, hit.astype(float), np.nan),
                           index=y_true.index)
    rolling = hit_series.rolling(window, min_periods=max(5, window // 2)).mean()

    fig, (ax_r, ax_c) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.3, 1]})

    # --- rolling hit-rate -----------------------------------------------------
    ax_r.plot(rolling.index, rolling, lw=1.5, label=f"{window}-day rolling hit-rate")
    ax_r.axhline(0.5, color="grey", ls="--", lw=1, label="coin flip (0.50)")
    ax_r.axhline(overall, color="C1", ls=":", lw=1.5,
                 label=f"overall = {overall:.1%}")
    # mark individual hits/misses along the bottom for texture
    ax_r.scatter(y_true.index[called & hit], np.full(int((called & hit).sum()), 0.02),
                 marker="|", color="green", s=40, alpha=0.5)
    ax_r.scatter(y_true.index[called & ~hit], np.full(int((called & ~hit).sum()), 0.0),
                 marker="|", color="red", s=40, alpha=0.5)
    ax_r.set_ylim(-0.03, 1.03)
    ax_r.set_ylabel("directional hit-rate")
    ax_r.set_title("Next-day directional accuracy (single-period, not compounded)")
    ax_r.legend(loc="upper left", fontsize=9)

    # --- confusion matrix (up/down) ------------------------------------------
    pred_up = yp[called] > 0
    act_up = yt[called] > 0
    cm = np.array([
        [np.sum(pred_up & act_up),   np.sum(pred_up & ~act_up)],    # predicted up
        [np.sum(~pred_up & act_up),  np.sum(~pred_up & ~act_up)],   # predicted down
    ])
    ax_c.imshow(cm, cmap="Blues")
    ax_c.set_xticks([0, 1], ["actual ↑", "actual ↓"])
    ax_c.set_yticks([0, 1], ["pred ↑", "pred ↓"])
    for (i, j), val in np.ndenumerate(cm):
        ax_c.text(j, i, str(int(val)), ha="center", va="center",
                  color="white" if val > cm.max() / 2 else "black", fontsize=13)
    ax_c.set_title("Up/down confusion matrix")

    fig.tight_layout()
    plt.close(fig)
    return fig
