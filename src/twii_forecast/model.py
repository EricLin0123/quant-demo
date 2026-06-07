"""Single LightGBM regressor: random-search tuning + early stopping on validation.

Objective is ``regression_l1`` (MAE) by default — daily returns are leptokurtic, so
an L1 loss is more robust to tail days than L2. We report both MAE and RMSE downstream
regardless. A small random search (config.N_RANDOM_SEARCH draws) over the LightGBM
hyper-parameters picks the config with the lowest validation MAE; the winner is then
refit on train with early stopping. Everything is seeded from ``config.SEED``.
"""

from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config

logger = logging.getLogger(__name__)


def _sample_params(rng: np.random.Generator) -> dict:
    s = config.SEARCH_SPACE

    def loguniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    return {
        "num_leaves": int(rng.integers(s["num_leaves"][0], s["num_leaves"][1] + 1)),
        "learning_rate": loguniform(*s["learning_rate"]),
        "max_depth": int(rng.integers(s["max_depth"][0], s["max_depth"][1] + 1)),
        "min_child_samples": int(
            rng.integers(s["min_child_samples"][0], s["min_child_samples"][1] + 1)
        ),
        "feature_fraction": float(rng.uniform(*s["feature_fraction"])),
        "bagging_fraction": float(rng.uniform(*s["bagging_fraction"])),
        "lambda_l1": loguniform(*s["lambda_l1"]),
        "lambda_l2": loguniform(*s["lambda_l2"]),
    }


def _base_params(extra: dict | None = None) -> dict:
    params = {
        "objective": config.OBJECTIVE,
        "metric": "l1",
        "n_estimators": config.N_ESTIMATORS,
        "seed": config.SEED,
        "bagging_freq": 1,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if extra:
        params.update(extra)
    return params


def _fit_one(
    params: dict,
    Xtr: pd.DataFrame, ytr: pd.Series,
    Xva: pd.DataFrame, yva: pd.Series,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def _val_mae(model: lgb.LGBMRegressor, Xva: pd.DataFrame, yva: pd.Series) -> float:
    pred = model.predict(Xva, num_iteration=model.best_iteration_)
    return float(np.mean(np.abs(yva.values - pred)))


def random_search(
    Xtr: pd.DataFrame, ytr: pd.Series,
    Xva: pd.DataFrame, yva: pd.Series,
    n_iter: int = config.N_RANDOM_SEARCH,
) -> tuple[lgb.LGBMRegressor, dict, float]:
    """Random search over the LightGBM space; return (best_model, best_params, best_mae)."""
    rng = np.random.default_rng(config.SEED)
    best_model, best_params, best_mae = None, None, np.inf

    for _ in tqdm(range(n_iter), desc="random search"):
        cand = _sample_params(rng)
        model = _fit_one(_base_params(cand), Xtr, ytr, Xva, yva)
        mae = _val_mae(model, Xva, yva)
        if mae < best_mae:
            best_model, best_params, best_mae = model, cand, mae

    logger.info("best val MAE=%.6e with params=%s", best_mae, best_params)
    return best_model, best_params, best_mae


def train(
    Xtr: pd.DataFrame, ytr: pd.Series,
    Xva: pd.DataFrame, yva: pd.Series,
    n_iter: int = config.N_RANDOM_SEARCH,
    tune: bool = True,
) -> tuple[lgb.LGBMRegressor, dict]:
    """Tune (optional) then refit the winning config with early stopping.

    Returns (fitted_model, best_params). Persists the model + params to disk.
    """
    if tune:
        _, best_params, _ = random_search(Xtr, ytr, Xva, yva, n_iter=n_iter)
    else:
        best_params = {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 6}

    model = _fit_one(_base_params(best_params), Xtr, ytr, Xva, yva)
    logger.info("refit best_iteration=%s", model.best_iteration_)

    model.booster_.save_model(
        str(config.MODEL_PATH), num_iteration=model.best_iteration_
    )
    payload = {**best_params, "best_iteration": int(model.best_iteration_)}
    config.PARAMS_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("saved model -> %s, params -> %s", config.MODEL_PATH, config.PARAMS_PATH)
    return model, best_params


def predict(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X, num_iteration=model.best_iteration_)
