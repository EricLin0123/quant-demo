"""Train-vs-test data-drift report via evidently — the MLOps touch.

Markets are non-stationary, so the test regime almost always differs from train.
Surfacing that drift is context for interpreting the test metrics, not a failure
condition. Falls back gracefully across evidently's API versions.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def drift_report(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 html_path=None) -> object | None:
    """Generate an evidently DataDrift report; save HTML; return the report obj."""
    html_path = html_path or (config.REPORTS_DIR / "drift_train_vs_test.html")

    # evidently >=0.7 (Report + DataDriftPreset under evidently.*)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=X_train, current_data=X_test)
        try:
            result.save_html(str(html_path))
        except Exception:  # noqa: BLE001 — older save signature
            report.save_html(str(html_path))
        logger.info("Saved drift report -> %s", html_path)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidently new API path failed (%s); trying legacy API", exc)

    # Legacy evidently (<0.7): evidently.report.Report + metric_preset
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_train, current_data=X_test)
        report.save_html(str(html_path))
        logger.info("Saved drift report (legacy API) -> %s", html_path)
        return report
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not generate evidently drift report: %s", exc)
        return None
