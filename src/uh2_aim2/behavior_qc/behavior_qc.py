#!/usr/bin/env python
"""
Behavioral QC Pipeline for UH2 AIM2.

Runs quality control on behavioral data and outputs:
- Per-task QC CSVs
- Combined exclusion file (subject_id, task, metric, metric_value, threshold)
"""

import os
from glob import glob
from pathlib import Path

import pandas as pd

from src.uh2_aim2.config import (
    BEHAVIOR_QC_PATH,
    EVENT_FILES_PATH,
    TASKS,
)
from src.uh2_aim2.utils.behavior_exclusion_utils import (
    run_all_exclusion_checks,
    summarize_exclusions,
)
from src.uh2_aim2.utils.behavior_qc_utils import (
    compute_acc_summary,
    compute_rt_summary,
    format_qc_results,
)


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete behavioral QC pipeline.

    Returns:
        Tuple of (qc_df, exclusion_df)
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UH2 AIM2 Behavioral QC Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    task_data = load_task_data(event_files_path)

    # Compute RT QC
    print("\n[2/4] Computing RT metrics...")
    rt_qc = {task: compute_rt_summary(df, task) for task, df in task_data.items()}
    rt_df = format_qc_results(rt_qc)

    # Compute accuracy QC
    print("\n[3/4] Computing accuracy metrics...")
    acc_qc = {task: compute_acc_summary(df, task) for task, df in task_data.items()}
    acc_df = format_qc_results(acc_qc)

    # Run exclusion checks
    print("\n[4/4] Running exclusion checks...")
    exclusion_df = run_all_exclusion_checks(acc_df)

    # Save outputs
    print("\n" + "=" * 60)
    print(f"Saving outputs to: {output_path}")
    print("=" * 60)

    # Per-task QC CSVs
    for task, df in acc_qc.items():
        path = os.path.join(output_path, f"{task}_qc.csv")
        df.to_csv(path)
        print(f"  {task}_qc.csv")

    # Combined QC files
    acc_df.to_csv(os.path.join(output_path, "all_tasks_acc_qc.csv"))
    rt_df.to_csv(os.path.join(output_path, "all_tasks_rt_qc.csv"))
    print("  all_tasks_acc_qc.csv")
    print("  all_tasks_rt_qc.csv")

    # Exclusions
    exclusion_df.to_csv(os.path.join(output_path, "exclusions.csv"), index=False)
    print("  exclusions.csv")

    # Summary
    summary = summarize_exclusions(exclusion_df)
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total subjects: {len(acc_df) - 2}")  # Subtract mean/std rows
    print(f"  Total exclusions: {summary.get('total_exclusions', 0)}")
    print(f"  Subjects with exclusions: {summary.get('unique_subjects_with_exclusions', 0)}")

    if "exclusions_by_task" in summary:
        print("\n  By task:")
        for task, count in summary["exclusions_by_task"].items():
            print(f"    {task}: {count}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return acc_df, exclusion_df


if __name__ == "__main__":
    run_qc_pipeline()
