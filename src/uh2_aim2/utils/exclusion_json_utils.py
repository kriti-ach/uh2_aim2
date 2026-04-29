"""Build unified ``exclusions.json`` payloads (behavioral, fMRIPrep, subject-wide spillover)."""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from uh2_aim2.config import (
    BEHAVIOR_EXCLUSIONS_TIMING_ALLOWLIST,
    BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD,
    EXCLUSION_REASON_EF_TIMING_OFF,
    EXCLUSION_REASON_MISSING_BEHAVIOR_FILE,
    EXCLUSION_REASON_SUBJECT_WIDE_OTHER,
    FULL_QC_CANONICAL_TASKS,
    FMRIPREP_FD_MEAN_INCLUDE_THRESHOLD_MM,
    SUBJECT_WIDE_EXCLUSION_THRESHOLD,
)
from uh2_aim2.utils.behavior_qc_utils import standardize_subject_numbers


def normalize_subject_bids(subject_value: object) -> str:
    """BIDS-style subject id: ``sub-1021``."""
    raw = str(subject_value).strip()
    if raw.startswith("sub-"):
        token = raw.replace("sub-", "")
    elif raw.startswith("s"):
        token = raw[1:]
    else:
        token = raw
    token = token.lstrip("0") or "0"
    return f"sub-{token}"


_REASON_PRIORITY = {
    "behavioral exclusion": 0,
    EXCLUSION_REASON_EF_TIMING_OFF: 1,
    EXCLUSION_REASON_MISSING_BEHAVIOR_FILE: 2,
}


def pair_in_behavior_exclusions_allowlist(subject_id: object, task: object) -> bool:
    sid_for_std = subject_id if isinstance(subject_id, (str, int)) else str(subject_id)
    s_int = standardize_subject_numbers(sid_for_std)
    t_str = str(task).strip()
    for sid_a, task_a in BEHAVIOR_EXCLUSIONS_TIMING_ALLOWLIST:
        if int(sid_a) == int(s_int) and str(task_a) == t_str:
            return True
    return False


def timing_rows_for_exclusions_json(timing_df: pd.DataFrame) -> pd.DataFrame:
    """Same flag logic as ``qc_behavior_timings`` flagged CSV, plus unreadable CSV rows."""
    if timing_df.empty:
        return timing_df
    fr = timing_df["flag_reason"].astype(str)
    delta_ok = pd.to_numeric(timing_df["delta"], errors="coerce")
    thresh = float(BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD)
    mask = (~timing_df["ok"]) & (
        fr.eq("missing csv")
        | (delta_ok >= thresh)
        | fr.str.startswith("read error")
    )
    return timing_df.loc[mask]


def merge_behavioral_exclusion_json_records(
    exclusion_df: pd.DataFrame,
    missing_raw_df: pd.DataFrame,
    timing_df: pd.DataFrame,
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}

    def upsert(subject_value: object, task_value: object, reason: str) -> None:
        key_sub = normalize_subject_bids(subject_value)
        task_s = str(task_value).strip()
        key = (key_sub, task_s)
        pri = _REASON_PRIORITY.get(reason, 99)
        if key not in merged or pri < merged[key][0]:
            merged[key] = (pri, {"subject": key_sub, "task": task_s, "reason": reason})

    if not exclusion_df.empty:
        pairs = exclusion_df[["subject_id", "task"]].drop_duplicates()
        for _, row in pairs.iterrows():
            upsert(row["subject_id"], row["task"], "behavioral exclusion")

    timing_sub = timing_rows_for_exclusions_json(timing_df)
    if not timing_sub.empty:
        for _, row in timing_sub.iterrows():
            if pair_in_behavior_exclusions_allowlist(row["subject_id"], row["task"]):
                continue
            fr = str(row["flag_reason"]).strip()
            reason = (
                EXCLUSION_REASON_MISSING_BEHAVIOR_FILE
                if fr == "missing csv"
                else EXCLUSION_REASON_EF_TIMING_OFF
            )
            upsert(row["subject_id"], row["task"], reason)

    if not missing_raw_df.empty:
        for _, row in missing_raw_df.iterrows():
            if pair_in_behavior_exclusions_allowlist(row["subject_id"], row["task"]):
                continue
            upsert(row["subject_id"], row["task"], EXCLUSION_REASON_MISSING_BEHAVIOR_FILE)

    return [
        pair[1]
        for pair in sorted(
            merged.values(),
            key=lambda x: (x[1]["subject"], x[1]["task"]),
        )
    ]


def fmriprep_exclusion_records_from_metrics(metrics_df: pd.DataFrame) -> list[dict[str, str]]:
    """
    fMRIPrep exclusion rows (same rules as ``qc_fmriprep_reports``):

    - ``rest``: exclude when FD mean > threshold.
    - non-rest: exclude when ``high_motion_flag``.
    """
    if metrics_df.empty:
        return []

    exclusions: list[dict[str, str]] = []
    for _, row in metrics_df.iterrows():
        task = str(row.get("task", "")).strip()
        task_lower = task.lower()
        fd_mean = row.get("fd_mean_mm", float("nan"))
        try:
            fd_mean_f = float(fd_mean)
        except (TypeError, ValueError):
            fd_mean_f = float("nan")
        high_motion = bool(row.get("high_motion_flag", False))

        reason: str | None = None
        if task_lower == "rest":
            if fd_mean_f > float(FMRIPREP_FD_MEAN_INCLUDE_THRESHOLD_MM):
                reason = "Subject had FD mean > 0.2mm"
        else:
            if high_motion:
                reason = (
                    "Subject had more than 20% of TRs with FD > 0.5mm or DVARS > 1.5"
                )

        if reason is None:
            continue

        exclusions.append(
            {
                "subject": normalize_subject_bids(row.get("subject_id")),
                "task": task,
                "reason": reason,
            }
        )

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in exclusions:
        key = (item["subject"], item["task"])
        if key not in unique:
            unique[key] = item

    return sorted(unique.values(), key=lambda x: (x["subject"], x["task"]))


def canonical_task_key(task_str: str, canonical: tuple[str, ...]) -> str | None:
    """Return canonical task name if ``task_str`` matches one of ``canonical`` (case-insensitive)."""
    t = str(task_str).strip()
    for c in canonical:
        if t.lower() == c.lower():
            return c
    return None


def compute_subject_wide_other_exclusions(
    behavioral_exclusions: list[dict[str, str]],
    fmriprep_exclusions: list[dict[str, str]],
    *,
    canonical_tasks: tuple[str, ...] = FULL_QC_CANONICAL_TASKS,
) -> list[dict[str, str]]:
    """
    If a subject has **more than** ``SUBJECT_WIDE_EXCLUSION_THRESHOLD`` excluded tasks
    among ``canonical_tasks`` (union of behavioral + fMRIPrep), add every remaining
    canonical task for that subject under ``other_exclusions``.
    """
    min_needed_excluded = int(SUBJECT_WIDE_EXCLUSION_THRESHOLD) + 1

    per_subject: dict[str, set[str]] = defaultdict(set)

    for row in behavioral_exclusions + fmriprep_exclusions:
        sub = normalize_subject_bids(row["subject"])
        ct = canonical_task_key(row["task"], canonical_tasks)
        if ct is not None:
            per_subject[sub].add(ct)

    spill: list[dict[str, str]] = []
    for subj, excluded in per_subject.items():
        if len(excluded) < min_needed_excluded:
            continue
        for task in canonical_tasks:
            if task not in excluded:
                spill.append(
                    {
                        "subject": subj,
                        "task": task,
                        "reason": EXCLUSION_REASON_SUBJECT_WIDE_OTHER,
                    }
                )

    return sorted(spill, key=lambda x: (x["subject"], x["task"]))


def write_unified_exclusions_json(
    json_path: str,
    *,
    behavioral_exclusions: list[dict[str, str]],
    fmriprep_exclusions: list[dict[str, str]],
    other_exclusions: list[dict[str, str]],
) -> None:
    """Write ``exclusions.json`` with the three standard sections."""
    from pathlib import Path

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "behavioral_exclusions": behavioral_exclusions,
        "fmriprep_exclusions": fmriprep_exclusions,
        "other_exclusions": other_exclusions,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def write_behavioral_only_to_exclusions_json(
    exclusion_df: pd.DataFrame,
    missing_raw_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    json_path: str,
) -> None:
    """
    Update only ``behavioral_exclusions``, preserving ``fmriprep_exclusions`` and
    ``other_exclusions`` from an existing file when present.
    """
    from pathlib import Path

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}

    payload.setdefault("behavioral_exclusions", [])
    payload.setdefault("fmriprep_exclusions", [])
    payload.setdefault("other_exclusions", [])

    payload["behavioral_exclusions"] = merge_behavioral_exclusion_json_records(
        exclusion_df, missing_raw_df, timing_df
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
