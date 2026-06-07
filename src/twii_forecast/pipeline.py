"""End-to-end orchestration so the notebook stays thin.

Order (matches plan §13):
  data -> features/target/dataset -> temporal-consistency selection (train only)
  -> chronological split -> scale (fit on train) -> LightGBM tune+refit
  -> per-next-day evaluation vs baselines -> SHAP.

Returns a single results dict the notebook renders.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import (
    config,
    data,
    dataset,
    evaluate,
    explain,
    feature_selection,
    model as model_mod,
    monitor,
    scaling,
    split as split_mod,
)

logger = logging.getLogger(__name__)


def run(
    n_iter: int = config.N_RANDOM_SEARCH,
    tune: bool = True,
    use_cache: bool = True,
) -> dict:
    # 1. data + 2. dataset
    df_ohlcv, volume_ok = data.load(use_cache=use_cache)
    table, feature_cols = dataset.build_dataset(df_ohlcv, volume_ok, use_cache=use_cache)

    # 3. split (do this before feature selection so selection sees train only)
    sp = split_mod.chronological_split(table)

    # 4. temporal consistency on TRAIN ONLY
    kept, auc_table = feature_selection.temporal_consistency(
        sp.train[feature_cols], feature_cols
    )
    if not kept:
        logger.warning("temporal consistency dropped everything; falling back to all features")
        kept = feature_cols

    # 4b. drift report on the kept (raw) features — diagnostic, not a gate
    drift_path = monitor.drift_report(sp.train[kept], sp.test[kept])

    # 5. scale (fit on train only), features only
    scaler = scaling.fit_scaler(sp.train[kept])
    Xtr = scaling.apply_scaler(scaler, sp.train[kept])
    Xva = scaling.apply_scaler(scaler, sp.val[kept])
    Xte = scaling.apply_scaler(scaler, sp.test[kept])
    ytr, yva, yte = (sp.train[dataset.TARGET_COL],
                     sp.val[dataset.TARGET_COL],
                     sp.test[dataset.TARGET_COL])

    # 6. train
    model, best_params = model_mod.train(Xtr, ytr, Xva, yva, n_iter=n_iter, tune=tune)

    # 7. evaluate — per next-day prediction, vs baselines
    y_pred = model_mod.predict(model, Xte)
    results_table = evaluate.evaluate(yte.values, y_pred, ytr.values)

    # 8. SHAP importance on the test matrix
    sv = explain.shap_values(model, Xte)
    importance = explain.mean_abs_importance(sv)

    return {
        "df_ohlcv": df_ohlcv,
        "volume_ok": volume_ok,
        "table": table,
        "feature_cols": feature_cols,
        "kept_features": kept,
        "auc_table": auc_table,
        "drift_report": drift_path,
        "split": sp,
        "scaler": scaler,
        "Xte": Xte,
        "y_test": yte,
        "y_pred": pd.Series(y_pred, index=yte.index, name="pred"),
        "y_train": ytr,
        "model": model,
        "best_params": best_params,
        "metrics_table": results_table,
        "shap_values": sv,
        "importance": importance,
    }
