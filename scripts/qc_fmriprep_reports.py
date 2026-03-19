#!/usr/bin/env python
"""Generate scan-level fMRIPrep motion QC CSV from confounds TSV files."""

from __future__ import annotations

from pathlib import Path

from uh2_aim2.config import (
    FMRIPREP_DERIVATIVES_PATH,
    FMRIPREP_DVARS_TR_THRESHOLD,
    FMRIPREP_FD_TR_THRESHOLD_MM,
    FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD,
    FMRIPREP_QC_OUTPUT_CSV,
    FMRIPREP_QC_OUTPUT_DIR,
)
from uh2_aim2.utils.fmriprep_qc_utils import collect_fmriprep_motion_metrics


def main() -> None:
    output_dir = Path(FMRIPREP_QC_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = collect_fmriprep_motion_metrics(
        fmriprep_root=FMRIPREP_DERIVATIVES_PATH,
        fd_threshold_mm=FMRIPREP_FD_TR_THRESHOLD_MM,
        std_dvars_threshold=FMRIPREP_DVARS_TR_THRESHOLD,
        high_motion_threshold_percent=FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD,
    )

    metrics_df.to_csv(FMRIPREP_QC_OUTPUT_CSV, index=False)
    print(f"Saved fMRIPrep QC metrics: {FMRIPREP_QC_OUTPUT_CSV}")
    print(f"Rows saved: {len(metrics_df)}")
    if not metrics_df.empty:
        print(
            f"Rows with >{FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD}% motion spikes: "
            f"{int(metrics_df['high_motion_flag'].sum())}"
        )


if __name__ == "__main__":
    main()