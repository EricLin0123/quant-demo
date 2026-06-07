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
