"""Flag detection for behavioral QC using vectorized pandas operations.

Outputs flags in format: subject_id, task, metric, metric_value, threshold
"""

import numpy as np
import pandas as pd


def check_stop_failure_rt_greater_than_go_rt(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check if stop failure RT is greater than go RT for each subject."""
    flags = []

    for task in ["stopSignal", "motorSelectiveStop"]:
        # Define column names based on task
        if task == "stopSignal":
            stop_failure_rt_col = "stop_failure_rt"
            go_rt_col = "go_rt"
        else:  # motorSelectiveStop
            stop_failure_rt_col = "crit_stop_failure_rt"
            go_rt_col = "crit_go_rt"

        # Check if columns exist
        if stop_failure_rt_col not in qc_df.columns or go_rt_col not in qc_df.columns:
            continue

        # Exclude mean/std rows
        mask = ~qc_df.index.isin(["mean", "std"])
        subjects_df = qc_df.loc[mask]

        # Get subjects where stop_failure_rt > go_rt
        for subj in subjects_df.index:
            stop_failure_rt = subjects_df.loc[subj, stop_failure_rt_col]
            go_rt = subjects_df.loc[subj, go_rt_col]

            # Skip if either value is NaN
            if pd.isna(stop_failure_rt) or pd.isna(go_rt):
                continue

            # Flag if stop failure RT is greater than go RT
            if stop_failure_rt > go_rt:
                flags.append({
                    "subject_id": subj,
                    "task": task,
                    "metric": "stop_failure_rt_greater_than_go_rt",
                    "metric_value": stop_failure_rt,
                    "threshold": f"> go_rt ({go_rt:.3f})",
                })

    return pd.DataFrame(flags)


def run_all_flagging_checks(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Run all flagging checks and return consolidated flag DataFrame."""
    checks = [
        check_stop_failure_rt_greater_than_go_rt(qc_df),
    ]

    # Filter out empty DataFrames and concatenate
    valid_checks = [df for df in checks if not df.empty]
    
    if not valid_checks:
        return pd.DataFrame(columns=["subject_id", "task", "metric", "metric_value", "threshold"])
    
    return pd.concat(valid_checks, ignore_index=True)