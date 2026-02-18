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
)
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
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Run the complete behavioral QC pipeline.

    Returns:
        Tuple of (qc_dict, exclusion_df) where qc_dict maps task -> QC DataFrame
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UH2 AIM2 Behavioral QC Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    task_data = load_task_data(event_files_path)

    # Compute QC metrics (both RT and accuracy)
    print("\n[2/3] Computing QC metrics...")
    qc_results = {}
    for task, df in task_data.items():
        print(f"  Processing {task}...")
        qc_results[task] = compute_qc_summary(df, task)

    # Run exclusion checks (using the last two rows which are NOT mean/std)
    print("\n[3/3] Running exclusion checks...")
    # Combine all QC data for exclusion checks (exclude mean/std rows)
    all_qc_data = pd.concat([
        df.iloc[:-2].add_prefix(f"{task}_") 
        for task, df in qc_results.items()
    ], axis=1)
    exclusion_df = run_all_exclusion_checks(all_qc_data)

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

    # Summary
    summary = summarize_exclusions(exclusion_df)
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total subjects: {len(qc_results) - 2}")  # Subtract mean/std rows
    print(f"  Total exclusions: {summary.get('total_exclusions', 0)}")
    print(f"  Subjects with exclusions: {summary.get('unique_subjects_with_exclusions', 0)}")

    if "exclusions_by_task" in summary:
        print("\n  By task:")
        for task, count in summary["exclusions_by_task"].items():
            print(f"    {task}: {count}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return qc_results, exclusion_df

if __name__ == "__main__":
    run_qc_pipeline()