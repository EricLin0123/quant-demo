"""Train-vs-test data-drift report (evidently).

A whole-distribution complement to the per-feature temporal-consistency filter in
`feature_selection.py`: that one *selects* features by trimonthly AUC; this one
*reports* how the kept feature distributions shifted from the training period to the
held-out test window. Writes an HTML report to reports/.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def drift_report(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    fname: str = "drift_report.html",
) -> str | None:
    """Generate an evidently DataDrift report (train=reference, test=current).

    Defensive: if evidently's API differs, log and return None rather than break the
    pipeline (this is a diagnostic, not a correctness gate).
    """
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset

        cols = list(train_X.columns)
        data_def = DataDefinition(numerical_columns=cols)
        ref = Dataset.from_pandas(train_X[cols].reset_index(drop=True), data_definition=data_def)
        cur = Dataset.from_pandas(test_X[cols].reset_index(drop=True), data_definition=data_def)

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=ref, current_data=cur)

        out = config.REPORTS / fname
        snapshot.save_html(str(out))
        logger.info("wrote drift report -> %s", out)
        return str(out)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("evidently drift report skipped: %s", exc)
        return None
