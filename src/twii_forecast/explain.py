"""SHAP feature importance for the trained LightGBM (TreeSHAP, exact & fast).

TreeExplainer computes per-prediction Shapley values that exactly decompose each
forecast into base value + sum of feature contributions (local accuracy). We aggregate
them into the two standard global views: a mean-|SHAP| bar ranking and a beeswarm
distribution. The paper found `difference` / `c_minus_o` (both from C_t - O_t)
dominated — worth checking whether that holds for next-day *return* prediction here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def shap_values(model, X: pd.DataFrame):
    """Return a shap.Explanation for X using TreeExplainer."""
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer(X)


def mean_abs_importance(sv) -> pd.DataFrame:
    """Ranked mean |SHAP| per feature."""
    vals = np.abs(sv.values).mean(axis=0)
    df = pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": vals})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_plots(sv, prefix: str = "shap") -> dict[str, str]:
    """Write beeswarm + bar plots to reports/. Returns {name: path}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    paths = {}

    plt.figure()
    shap.plots.beeswarm(sv, show=False, max_display=20)
    plt.tight_layout()
    p = config.REPORTS / f"{prefix}_beeswarm.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    paths["beeswarm"] = str(p)

    plt.figure()
    shap.plots.bar(sv, show=False, max_display=20)
    plt.tight_layout()
    p = config.REPORTS / f"{prefix}_bar.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    paths["bar"] = str(p)

    logger.info("saved SHAP plots: %s", paths)
    return paths
