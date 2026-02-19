#!/usr/bin/env python
"""
Behavioral QC Pipeline for UH2 AIM2.

Runs quality control on behavioral data and outputs:
- Per-task QC CSVs (containing both RT and accuracy metrics)
- Combined exclusion file (subject_id, task, metric, metric_value, threshold)
"""

import os
from glob import glob
from pathlib import Path

import pandas as pd

from config import (
    BEHAVIOR_QC_PATH,
    EVENT_FILES_PATH,
    TASKS,
)
from utils.behavior_exclusion_utils import (
    run_all_exclusion_checks,
    summarize_exclusions,
    check_missing_data,
)
from utils.behavior_flagging_utils import run_all_flagging_checks
from utils.behavior_qc_utils import compute_qc_summary


def load_task_data(event_files_path: str = EVENT_FILES_PATH) -> dict[str, pd.DataFrame]:
    """Load behavioral event files for all tasks."""
    data = {}

    for task in TASKS:
        files = glob(os.path.join(event_files_path, f"*{task}*_events.tsv"))
        if files:
            data[task] = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
            print(f"  {task}: {len(files)} files, {len(data[task])} trials")
        else:
            print(f"  {task}: no files found")

    return data


def run_qc_pipeline(
    event_files_path: str = EVENT_FILES_PATH,
    output_path: str = BEHAVIOR_QC_PATH,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete behavioral QC pipeline.

    Returns:
        Tuple of (qc_dict, exclusion_df, flags_df, missing_df) where:
            - qc_dict maps task -> QC DataFrame
            - exclusion_df contains exclusion criteria violations
            - flags_df contains warnings/flags
            - missing_df contains subjects with missing task data
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UH2 AIM2 Behavioral QC Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    task_data = load_task_data(event_files_path)

    # Compute QC metrics (both RT and accuracy)
    print("\n[2/5] Computing QC metrics...")
    qc_results = {}
    for task, df in task_data.items():
        print(f"  Processing {task}...")
        qc_results[task] = compute_qc_summary(df, task)

    # Combine all QC data for exclusion/flagging checks (exclude mean/std rows)
    all_qc_data = pd.concat([
        df.iloc[:-2].add_prefix(f"{task}_") 
        for task, df in qc_results.items()
    ], axis=1)

    # Check for missing data
    print("\n[3/5] Checking for missing data...")
    missing_df = check_missing_data(all_qc_data)

    # Run exclusion checks (WITHOUT missing data check)
    print("\n[4/5] Running exclusion checks...")
    exclusion_df = run_all_exclusion_checks(all_qc_data, missing_df)

    # Run flagging checks
    print("\n[5/5] Running flagging checks...")
    flags_df = run_all_flagging_checks(all_qc_data)

    # Save outputs
    print("\n" + "=" * 60)
    print(f"Saving outputs to: {output_path}")
    print("=" * 60)

    # Per-task QC CSVs (with mean and std already included)
    for task, df in qc_results.items():
        path = os.path.join(output_path, f"{task}_qc.csv")
        df.to_csv(path)
        print(f"  {task}_qc.csv")

    # Exclusions
    exclusion_df.to_csv(os.path.join(output_path, "exclusions.csv"), index=False)
    print("  exclusions.csv")

    # Flags
    flags_df.to_csv(os.path.join(output_path, "flags.csv"), index=False)
    print("  flags.csv")

    # Missing data
    missing_df.to_csv(os.path.join(output_path, "missing_data.csv"), index=False)
    print("  missing_data.csv")

    # Summary
    summary = summarize_exclusions(exclusion_df)
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    # Count subjects properly (first QC result, excluding mean/std)
    first_task_qc = next(iter(qc_results.values()))
    num_subjects = len(first_task_qc) - 2  # Subtract mean/std rows
    
    print(f"  Total subjects: {num_subjects}")
    print(f"  Total exclusions: {summary.get('total_exclusions', 0)}")
    print(f"  Subjects with exclusions: {summary.get('unique_subjects_with_exclusions', 0)}")
    print(f"  Total flags: {len(flags_df)}")
    print(f"  Subjects with flags: {flags_df['subject_id'].nunique() if not flags_df.empty else 0}")
    print(f"  Subjects with missing data: {missing_df['subject_id'].nunique() if not missing_df.empty else 0}")

    if "exclusions_by_task" in summary:
        print("\n  Exclusions by task:")
        for task, count in summary["exclusions_by_task"].items():
            print(f"    {task}: {count}")

    if not flags_df.empty:
        print("\n  Flags by task:")
        for task, count in flags_df.groupby("task").size().items():
            print(f"    {task}: {count}")

    if not missing_df.empty:
        print("\n  Missing data by task:")
        for task, count in missing_df.groupby("task").size().items():
            print(f"    {task}: {count}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return qc_results, exclusion_df, flags_df, missing_df


if __name__ == "__main__":
    run_qc_pipeline()