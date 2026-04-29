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
from uh2_aim2.utils.exclusion_json_utils import fmriprep_exclusion_records_from_metrics
from uh2_aim2.utils.fmriprep_qc_utils import collect_fmriprep_motion_metrics


def _update_final_exclusions_json(metrics_df, json_path: str) -> int:
    """Replace ``fmriprep_exclusions`` in exclusions.json; preserve other sections."""
    path = Path(json_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}

    payload.setdefault("behavioral_exclusions", [])
    payload.setdefault("fmriprep_exclusions", [])
    payload.setdefault("other_exclusions", [])

    payload["fmriprep_exclusions"] = fmriprep_exclusion_records_from_metrics(metrics_df)

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
