"""Single LightGBM regressor: baseline -> random search on val -> refit.

Objective is L1 (MAE): daily returns are leptokurtic / fat-tailed, so an L1 loss
is more robust to tail days than L2. We still report both MAE and RMSE.

A single chronological split (no walk-forward / CPCV) — the agreed simplification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# Random-search space. Ranges are deliberately conservative for a small, noisy
# financial dataset (shallow trees, strong regularisation).
SEARCH_SPACE = {
    "num_leaves": [7, 15, 31, 63],
    "learning_rate": (0.005, 0.1),          # log-uniform
    "max_depth": [3, 4, 5, 6, -1],
    "min_child_samples": [10, 20, 40, 80, 120],
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "lambda_l1": (0.0, 5.0),
    "lambda_l2": (0.0, 5.0),
}

BASE_PARAMS = {
    "objective": config.OBJECTIVE,
    "n_estimators": config.N_ESTIMATORS,
    "random_state": config.SEED,
    "n_jobs": -1,
    "verbose": -1,
    # bagging_fraction only takes effect when bagging_freq >= 1 — without this
    # LightGBM silently skips row subsampling and the tuned bagging_fraction
    # becomes a dead knob. Resample every iteration.
    "bagging_freq": 1,
}


@dataclass
class FitResult:
    model: lgb.LGBMRegressor
    params: dict
    best_iteration: int
    val_mae: float
    search_history: list[dict] = field(default_factory=list)


def _sample_params(rng: np.random.Generator) -> dict:
    p = {}
    for k, v in SEARCH_SPACE.items():
        if isinstance(v, list):
            p[k] = v[rng.integers(len(v))]
        else:  # (lo, hi) continuous
            lo, hi = v
            if k == "learning_rate":  # log-uniform
                p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                p[k] = float(rng.uniform(lo, hi))
    return p


def _fit_one(params: dict, X_tr, y_tr, X_va, y_va) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**{**BASE_PARAMS, **params})
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def _val_mae(model: lgb.LGBMRegressor, X_va, y_va) -> float:
    pred = model.predict(X_va, num_iteration=model.best_iteration_)
    return float(np.mean(np.abs(y_va.to_numpy() - pred)))


def train(X_tr, y_tr, X_va, y_va, n_draws: int = config.N_SEARCH_DRAWS,
          progress: bool = True) -> FitResult:
    """Random-search hyperparameters, select lowest val MAE, return best model."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **_):
            return x

    rng = np.random.default_rng(config.SEED)

    # Draw 0 is a sensible fixed baseline config so search never does worse.
    candidates = [{
        "num_leaves": 31, "learning_rate": 0.05, "max_depth": -1,
        "min_child_samples": 20, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "lambda_l1": 0.0, "lambda_l2": 0.0,
    }]
    candidates += [_sample_params(rng) for _ in range(n_draws - 1)]

    best: FitResult | None = None
    history: list[dict] = []
    it = tqdm(candidates, desc="random search") if progress else candidates
    for params in it:
        model = _fit_one(params, X_tr, y_tr, X_va, y_va)
        mae = _val_mae(model, X_va, y_va)
        history.append({**params, "val_mae": mae, "best_iter": model.best_iteration_})
        if best is None or mae < best.val_mae:
            best = FitResult(model=model, params=params,
                             best_iteration=model.best_iteration_, val_mae=mae)

    assert best is not None
    best.search_history = history
    logger.info("Best val MAE=%.6e at %d trees: %s",
                best.val_mae, best.best_iteration, best.params)
    return best


def save(result: FitResult) -> None:
    result.model.booster_.save_model(
        str(config.MODEL_PATH), num_iteration=result.best_iteration)
    payload = {
        "params": result.params,
        "best_iteration": result.best_iteration,
        "val_mae": result.val_mae,
    }
    config.PARAMS_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Saved model -> %s and params -> %s",
                config.MODEL_PATH, config.PARAMS_PATH)


def predict(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X, num_iteration=model.best_iteration_)
