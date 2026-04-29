"""QC for scanner/trigger wait durations in raw behavioral task CSVs."""

from __future__ import annotations

import os
from glob import glob
from typing import Any

import numpy as np
import pandas as pd

from uh2_aim2.config import (
    BEHAVIOR_DATA_UNPROCESSED,
    FMRI_TRIGGER_TR_MS,
    FMRI_TRIGGER_WAIT_DURATION_S,
    FMRI_TRIGGER_WAIT_TASKS,
    FMRI_TRIGGER_WAIT_TRIAL_ID,
    MANIPULATION_SCANNER_WAIT_DURATION_S,
    MANIPULATION_SCANNER_WAIT_TRIAL_ID,
    SECONDS_TO_MILLISECONDS,
    TASKS,
)
from uh2_aim2.utils.behavior_qc_utils import remove_practice_stage_rows


def _task_csv_path(subject: str, task: str) -> str:
    """Flat processed layout: ``{subject}_{task}_cleaned.csv`` under ``BEHAVIOR_DATA``."""
    return os.path.join(BEHAVIOR_DATA_UNPROCESSED, f"{subject}_{task}_cleaned.csv")


def _wait_span_seconds(
    df: pd.DataFrame, trial_id_col: str, time_col: str, trial_token: str
) -> tuple[float | None, str | None]:
    """
    Return span in **seconds** for rows whose trial_id matches ``trial_token``.

    - ``scanner_wait``: (max - min) ``time_elapsed`` (ms in CSV) → s.
    - ``fmri_trigger_wait``: file order, last minus **second** ``time_elapsed`` (ms)
      plus ``FMRI_TRIGGER_TR_MS``; single matching row uses ``block_duration`` (ms).
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
        second_te = numeric.iloc[1]
        last_te = numeric.iloc[-1]
        if pd.isna(second_te) or pd.isna(last_te):
            return None, "time_elapsed not numeric on second or last fmri_trigger_wait row"
        span_ms = float(last_te - second_te) + float(FMRI_TRIGGER_TR_MS)
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


def run_behavior_timing_qc(
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each subject × task, check wait-window span against config expectations.

    If ``subjects`` is None, infers subjects from ``*_cleaned.csv`` files in
    ``BEHAVIOR_DATA``. ``tasks`` defaults to ``TASKS``.
    """
    task_list = list(tasks) if tasks is not None else list(TASKS)

    if subjects is None:
        subjects = []
        if os.path.isdir(BEHAVIOR_DATA_UNPROCESSED):
            cleaned_files = glob(os.path.join(BEHAVIOR_DATA_UNPROCESSED, "*.csv"))
            found: set[str] = set()
            for path in cleaned_files:
                stem = os.path.basename(path).replace("*.csv", "")
                if "_" not in stem:
                    continue
                subj = stem.split("_", 1)[0]
                found.add(subj)
            subjects = sorted(found)

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
                df = remove_practice_stage_rows(pd.read_csv(path))
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

    return pd.DataFrame(rows)
