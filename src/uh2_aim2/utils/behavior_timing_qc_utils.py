"""QC for scanner/trigger wait durations in raw behavioral task CSVs."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from uh2_aim2.config import (
    BEHAVIOR_DATA_RAW,
    FMRI_TRIGGER_TR_MS,
    FMRI_TRIGGER_WAIT_DURATION_S,
    FMRI_TRIGGER_WAIT_TASKS,
    FMRI_TRIGGER_WAIT_TRIAL_ID,
    MANIPULATION_SCANNER_WAIT_DURATION_S,
    MANIPULATION_SCANNER_WAIT_TRIAL_ID,
    SECONDS_TO_MILLISECONDS,
    TASKS,
)


def _task_csv_path(subject: str, task: str) -> str:
    """Raw layout: ``BEHAVIOR_DATA_RAW/{subject}/{subject}_{task}.csv``."""
    return os.path.join(BEHAVIOR_DATA_RAW, subject, f"{subject}_{task}.csv")


def infer_subjects_from_behavior_raw() -> list[str]:
    """Infer subject ids from ``BEHAVIOR_DATA_RAW/{subject}`` directories."""
    return _infer_subjects_from_raw()


def _infer_subjects_from_raw() -> list[str]:
    """Infer subject ids from ``BEHAVIOR_DATA_RAW/{subject}`` directories."""
    if not os.path.isdir(BEHAVIOR_DATA_RAW):
        return []
    subjects: list[str] = []
    for entry in sorted(os.listdir(BEHAVIOR_DATA_RAW)):
        subj_dir = os.path.join(BEHAVIOR_DATA_RAW, entry)
        if os.path.isdir(subj_dir):
            subjects.append(entry)
    return subjects


def _wait_span_seconds(
    df: pd.DataFrame, trial_id_col: str, time_col: str, trial_token: str
) -> tuple[float | None, str | None]:
    """
    Return span in **seconds** for rows whose trial_id matches ``trial_token``.

    - ``scanner_wait``: (max - min) ``time_elapsed`` (ms in CSV) → s.
    - ``fmri_trigger_wait``: file order, last minus **first** ``time_elapsed`` (ms);
      single matching row uses ``block_duration`` (ms).
      For the three fmri-trigger tasks, ``run_behavior_timing_qc`` adds +1 TR only when
      this raw span is **not** already in the nominal band around ``expected`` (~10.88 s).
    """
    if trial_id_col not in df.columns:
        return None, f"missing column {trial_id_col!r}"
    if time_col not in df.columns:
        return None, f"missing column {time_col!r}"

    tid = df[trial_id_col].astype(str).str.strip()
    mask = tid == trial_token
    sub_rows = df.loc[mask]
    if sub_rows.empty:
        return None, f"no rows with {trial_id_col}=={trial_token!r}"

    if trial_token == FMRI_TRIGGER_WAIT_TRIAL_ID:
        if len(sub_rows) == 1:
            if "block_duration" not in sub_rows.columns:
                return None, "single wait row but missing column 'block_duration'"
            block_vals = pd.Series(
                pd.to_numeric(sub_rows["block_duration"], errors="coerce")
            )
            block_val = block_vals.iloc[0]
            if pd.isna(block_val):
                return None, "single wait row but block_duration not numeric"
            span_ms = float(block_val)
            span_s = span_ms / float(SECONDS_TO_MILLISECONDS)
            return span_s, None

        numeric = pd.Series(
            pd.to_numeric(pd.Series(sub_rows[time_col]), errors="coerce")
        )
        first_te = numeric.iloc[0]
        last_te = numeric.iloc[-1]
        if pd.isna(first_te) or pd.isna(last_te):
            return None, "time_elapsed not numeric on first or last fmri_trigger_wait row"
        span_ms = float(last_te - first_te)
        span_s = span_ms / float(SECONDS_TO_MILLISECONDS)
        return span_s, None

    sub = pd.Series(sub_rows[time_col])
    numeric = pd.Series(pd.to_numeric(sub, errors="coerce"))
    if numeric.isna().all():
        return None, "time_elapsed not numeric"

    span_ms = float(numeric.max() - numeric.min())
    span_s = span_ms / float(SECONDS_TO_MILLISECONDS)
    return span_s, None


def _expected_for_task(task: str) -> tuple[str, float] | None:
    if task == "manipulationTask":
        return MANIPULATION_SCANNER_WAIT_TRIAL_ID, MANIPULATION_SCANNER_WAIT_DURATION_S
    if task in FMRI_TRIGGER_WAIT_TASKS:
        return FMRI_TRIGGER_WAIT_TRIAL_ID, FMRI_TRIGGER_WAIT_DURATION_S
    return None


def _span_in_nominal_bucket(span_s: float, expected_s: float) -> bool:
    """True if span is in the hundredth-second band [expected, expected + 0.01)."""
    eps = 1e-9
    low, high = expected_s, expected_s + 0.01
    return (span_s + eps >= low) and (span_s < high)


def _already_near_expected_wait(span_s: float, expected_s: float) -> bool:
    """Raw span is already ~expected (e.g. ~10.88 s): do not add +1 TR for fmri-trigger tasks."""
    if _span_in_nominal_bucket(span_s, expected_s):
        return True
    return abs(float(span_s) - float(expected_s)) <= 0.01


def run_behavior_timing_qc(
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each subject × task, check wait-window span against config expectations.

    Uses each CSV **as-is** (no practice / experimentor_wait trimming).

    If ``subjects`` is None, infers subjects from subdirectories under
    ``BEHAVIOR_DATA_RAW``. ``tasks`` defaults to ``TASKS``.
    """
    result_columns = [
        "subject_id",
        "task",
        "trial_id_filter",
        "expected_duration",
        "observed_duration",
        "delta",
        "ok",
        "flag_reason",
    ]
    task_list = list(tasks) if tasks is not None else list(TASKS)

    if subjects is None:
        subjects = _infer_subjects_from_raw()

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
                        "trial_id_filter": trial_token,
                        "expected_duration": expected,
                        "observed_duration": np.nan,
                        "delta": np.nan,
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
                        "trial_id_filter": trial_token,
                        "expected_duration": expected,
                        "observed_duration": np.nan,
                        "delta": np.nan,
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
                        "trial_id_filter": trial_token,
                        "expected_duration": expected,
                        "observed_duration": np.nan,
                        "delta": np.nan,
                        "ok": False,
                        "flag_reason": err,
                    }
                )
                continue

            assert span is not None
            # For discountFix/motorSelectiveStop/stopSignal: add +1 TR (680ms) when
            # raw span is NOT already near the expected 10.88s.
            tr_sec = float(FMRI_TRIGGER_TR_MS) / float(SECONDS_TO_MILLISECONDS)
            if task in FMRI_TRIGGER_WAIT_TASKS and not _already_near_expected_wait(span, expected):
                span = span + tr_sec
            delta = abs(span - expected)
            ok = _span_in_nominal_bucket(span, expected)
            rows.append(
                {
                    "subject_id": subject,
                    "task": task,
                    "trial_id_filter": trial_token,
                    "expected_duration": expected,
                    "observed_duration": span,
                    "delta": delta,
                    "ok": ok,
                    "flag_reason": "" if ok else "duration mismatch",
                }
            )

    return pd.DataFrame(rows, columns=result_columns)
