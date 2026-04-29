#!/usr/bin/env python
"""
Run behavioral QC, fMRIPrep motion QC, and write one unified ``exclusions.json``.

Behavioral section: same as ``behavior_qc.py`` (metrics, missing raw CSVs, timing QC).
fMRIPrep section: same rules as ``qc_fmriprep_reports.py``.
``other_exclusions``: when a subject has more than two excluded tasks among the five
canonical tasks (motorSelectiveStop, stopSignal, discountFix, manipulationTask, rest),
every remaining canonical task for that subject is listed with a subject-wide reason.

Usage (from repo root)::

    PYTHONPATH=src python scripts/run_full_qc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import behavior_qc as behavior_qc_script  # noqa: E402

from uh2_aim2.config import (  # noqa: E402
    FINAL_EXCLUSIONS_JSON_PATH,
    FMRIPREP_DERIVATIVES_PATH,
    FMRIPREP_DVARS_TR_THRESHOLD,
    FMRIPREP_FD_MEAN_INCLUDE_THRESHOLD_MM,
    FMRIPREP_FD_TR_THRESHOLD_MM,
    FMRIPREP_HIGH_MOTION_TR_PERCENT_THRESHOLD,
    FMRIPREP_QC_OUTPUT_CSV,
    FMRIPREP_QC_OUTPUT_DIR,
)
from uh2_aim2.utils.exclusion_json_utils import (  # noqa: E402
    compute_subject_wide_other_exclusions,
    fmriprep_exclusion_records_from_metrics,
    merge_behavioral_exclusion_json_records,
    write_unified_exclusions_json,
)
from uh2_aim2.utils.fmriprep_qc_utils import collect_fmriprep_motion_metrics  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("UH2 AIM2 Full QC (behavioral + fMRIPrep → exclusions.json)")
    print("=" * 60)

    (
        _qc_results,
        exclusion_df,
        _flags_df,
        _missing_df,
        timing_df,
        missing_raw_df,
    ) = behavior_qc_script.run_qc_pipeline(skip_exclusions_json=True)

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
    print(f"\nSaved fMRIPrep QC metrics: {FMRIPREP_QC_OUTPUT_CSV}")
    print(f"  Rows: {len(metrics_df)}")

    behavioral_list = merge_behavioral_exclusion_json_records(
        exclusion_df, missing_raw_df, timing_df
    )
    fmriprep_list = fmriprep_exclusion_records_from_metrics(metrics_df)
    other_list = compute_subject_wide_other_exclusions(behavioral_list, fmriprep_list)

    write_unified_exclusions_json(
        FINAL_EXCLUSIONS_JSON_PATH,
        behavioral_exclusions=behavioral_list,
        fmriprep_exclusions=fmriprep_list,
        other_exclusions=other_list,
    )

    print(f"\nWrote unified exclusions: {FINAL_EXCLUSIONS_JSON_PATH}")
    print(
        f"  behavioral_exclusions: {len(behavioral_list)}, "
        f"fmriprep_exclusions: {len(fmriprep_list)}, "
        f"other_exclusions: {len(other_list)}"
    )


if __name__ == "__main__":
    main()
