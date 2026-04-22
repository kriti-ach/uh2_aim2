#!/usr/bin/env python
"""Generate scan-level fMRIPrep motion QC CSV from confounds TSV files."""

from __future__ import annotations

import json
from pathlib import Path

from uh2_aim2.config import (
    FINAL_EXCLUSIONS_JSON_PATH,
    FMRIPREP_DERIVATIVES_PATH,
    FMRIPREP_DVARS_TR_THRESHOLD,
    FMRIPREP_FD_MEAN_INCLUDE_THRESHOLD_MM,
    FMRIPREP_FD_TR_THRESHOLD_MM,
    FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD,
    FMRIPREP_QC_OUTPUT_CSV,
    FMRIPREP_QC_OUTPUT_DIR,
)
from uh2_aim2.utils.fmriprep_qc_utils import collect_fmriprep_motion_metrics


def _normalize_subject(subject_value: object) -> str:
    raw = str(subject_value).strip()
    if raw.startswith("sub-"):
        token = raw.replace("sub-", "")
    elif raw.startswith("s"):
        token = raw[1:]
    else:
        token = raw
    token = token.lstrip("0") or "0"
    return f"sub-{token}"


def _update_final_exclusions_json(metrics_df, json_path: str) -> int:
    """
    Update `fmriprep_exclusions` in exclusions.json (FINAL_EXCLUSIONS_JSON_PATH) from fMRIPrep QC metrics.

    Logic:
    - rest scans: exclude when fd_mean_mm > 0.2
    - non-rest scans: exclude when high_motion_flag is True
    """
    path = Path(json_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}

    payload.setdefault("behavioral_exclusions", [])
    payload.setdefault("fmriprep_exclusions", [])
    payload.setdefault("other_exclusions", [])

    if metrics_df.empty:
        payload["fmriprep_exclusions"] = []
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        return 0

    exclusions = []
    for _, row in metrics_df.iterrows():
        task = str(row.get("task", "")).strip()
        task_lower = task.lower()
        fd_mean = row.get("fd_mean_mm", float("nan"))
        high_motion = bool(row.get("high_motion_flag", False))

        reason = None
        if task_lower == "rest":
            if fd_mean > FMRIPREP_FD_MEAN_INCLUDE_THRESHOLD_MM:
                reason = "Subject had FD mean > 0.2mm"
        else:
            if high_motion:
                reason = "Subject had more than 20% of TRs with FD > 0.5mm or DVARS > 1.5"

        if reason is None:
            continue

        exclusions.append(
            {
                "subject": _normalize_subject(row.get("subject_id")),
                "task": task,
                "reason": reason,
            }
        )

    # Keep one entry per subject+task
    unique = {}
    for item in exclusions:
        key = (item["subject"], item["task"])
        if key not in unique:
            unique[key] = item

    payload["fmriprep_exclusions"] = list(unique.values())
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    return len(payload["fmriprep_exclusions"])


def main() -> None:
    output_dir = Path(FMRIPREP_QC_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = collect_fmriprep_motion_metrics(
        fmriprep_root=FMRIPREP_DERIVATIVES_PATH,
        fd_threshold_mm=FMRIPREP_FD_TR_THRESHOLD_MM,
        std_dvars_threshold=FMRIPREP_DVARS_TR_THRESHOLD,
        high_motion_threshold_percent=FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD,
        fd_mean_min=FMRIPREP_FD_MEAN_INCLUDE_THRESHOLD_MM,
    )

    metrics_df.to_csv(FMRIPREP_QC_OUTPUT_CSV, index=False)
    print(f"Saved fMRIPrep QC metrics: {FMRIPREP_QC_OUTPUT_CSV}")
    print(f"Rows saved: {len(metrics_df)}")
    if not metrics_df.empty:
        print(
            f"Rows with >{FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD}% motion spikes: "
            f"{int(metrics_df['high_motion_flag'].sum())}"
        )
    n_added = _update_final_exclusions_json(metrics_df, FINAL_EXCLUSIONS_JSON_PATH)
    print(f"Updated fmriprep exclusions in: {FINAL_EXCLUSIONS_JSON_PATH}")
    print(f"fmriprep exclusions count: {n_added}")


if __name__ == "__main__":
    main()