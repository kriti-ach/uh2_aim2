"""QC for scanner/trigger wait durations in raw behavioral task CSVs."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from uh2_aim2.config import (
    BEHAVIOR_DATA,
    BEHAVIOR_TIMING_TOLERANCE_S,
    FMRI_TRIGGER_WAIT_DURATION_S,
    FMRI_TRIGGER_WAIT_TASKS,
    FMRI_TRIGGER_WAIT_TRIAL_ID,
    MANIPULATION_SCANNER_WAIT_DURATION_S,
    MANIPULATION_SCANNER_WAIT_TRIAL_ID,
    TASKS,
)


def _task_csv_path(subject: str, task: str) -> str:
    return os.path.join(BEHAVIOR_DATA, subject, "task", f"{subject}_{task}.csv")


def _wait_span_seconds(
    df: pd.DataFrame, trial_id_col: str, time_col: str, trial_token: str
) -> tuple[float | None, str | None]:
    """
    Return (max - min) time_elapsed for rows whose trial_id matches trial_token.
    On failure, duration is None and the second value is a short reason.
    """
    if trial_id_col not in df.columns:
        return None, f"missing column {trial_id_col!r}"
    if time_col not in df.columns:
        return None, f"missing column {time_col!r}"

    tid = df[trial_id_col].astype(str).str.strip()
    mask = tid == trial_token
    sub = df.loc[mask, time_col]
    if sub.empty:
        return None, f"no rows with {trial_id_col}=={trial_token!r}"

    numeric = pd.to_numeric(sub, errors="coerce")
    if numeric.isna().all():
        return None, "time_elapsed not numeric"

    span = float(numeric.max() - numeric.min())
    return span, None


def _expected_for_task(task: str) -> tuple[str, float] | None:
    if task == "manipulationTask":
        return MANIPULATION_SCANNER_WAIT_TRIAL_ID, MANIPULATION_SCANNER_WAIT_DURATION_S
    if task in FMRI_TRIGGER_WAIT_TASKS:
        return FMRI_TRIGGER_WAIT_TRIAL_ID, FMRI_TRIGGER_WAIT_DURATION_S
    return None


def run_behavior_timing_qc(
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
    tolerance_s: float = BEHAVIOR_TIMING_TOLERANCE_S,
) -> pd.DataFrame:
    """
    For each subject × task, check wait-window span against config expectations.

    If ``subjects`` is None, uses every directory under ``BEHAVIOR_DATA`` that
    contains a ``task`` subdirectory. ``tasks`` defaults to ``TASKS``.
    """
    task_list = list(tasks) if tasks is not None else list(TASKS)

    if subjects is None:
        subjects = []
        if os.path.isdir(BEHAVIOR_DATA):
            for name in sorted(os.listdir(BEHAVIOR_DATA)):
                p = os.path.join(BEHAVIOR_DATA, name, "task")
                if os.path.isdir(p):
                    subjects.append(name)

    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for task in task_list:
            spec = _expected_for_task(task)
            if spec is None:
                continue
            trial_token, expected = spec
            path = _task_csv_path(subject, task)
            if not os.path.isfile(path):
                rows.append(
                    {
                        "subject_id": subject,
                        "task": task,
                        "csv_path": path,
                        "trial_id_filter": trial_token,
                        "expected_duration_s": expected,
                        "observed_duration_s": np.nan,
                        "delta_s": np.nan,
                        "ok": False,
                        "flag_reason": "missing csv",
                    }
                )
                continue

            try:
                df = pd.read_csv(path)
            except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                rows.append(
                    {
                        "subject_id": subject,
                        "task": task,
                        "csv_path": path,
                        "trial_id_filter": trial_token,
                        "expected_duration_s": expected,
                        "observed_duration_s": np.nan,
                        "delta_s": np.nan,
                        "ok": False,
                        "flag_reason": f"read error: {exc}",
                    }
                )
                continue

            span, err = _wait_span_seconds(
                df, "trial_id", "time_elapsed", trial_token
            )
            if err is not None:
                rows.append(
                    {
                        "subject_id": subject,
                        "task": task,
                        "csv_path": path,
                        "trial_id_filter": trial_token,
                        "expected_duration_s": expected,
                        "observed_duration_s": np.nan,
                        "delta_s": np.nan,
                        "ok": False,
                        "flag_reason": err,
                    }
                )
                continue

            delta = abs(span - expected)
            ok = bool(np.isclose(span, expected, rtol=0.0, atol=tolerance_s))
            rows.append(
                {
                    "subject_id": subject,
                    "task": task,
                    "csv_path": path,
                    "trial_id_filter": trial_token,
                    "expected_duration_s": expected,
                    "observed_duration_s": span,
                    "delta_s": delta,
                    "ok": ok,
                    "flag_reason": "" if ok else "duration mismatch",
                }
            )

    return pd.DataFrame(rows)
