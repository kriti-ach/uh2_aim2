"""Exclusion detection for behavioral QC using vectorized pandas operations.

Outputs exclusions in format: subject_id, task, metric, metric_value, threshold
"""

import os

import numpy as np
import pandas as pd

from config import (
    DISCOUNT_PROPORTION_MAX,
    DISCOUNT_PROPORTION_MIN,
    MIN_VALID_TASKS,
    MOTOR_STOP_NONCRIT_OMISSION_MAX,
    OMISSION_RATE_MAX,
    STOP_SUCCESS_MAX,
    STOP_SUCCESS_MIN,
    SUBJECT_DATA_PATH,
    SUBJECTIVE_EXCLUSIONS,
    TASKS,
    TRUNCATION_RATE_MAX,
    STOP_SIGNAL_GO_ACC,
    STOP_SIGNAL_GO_RT,
)


def check_stop_success_rate(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check stop success rate is within acceptable bounds."""
    exclusions = []

    for task in ["stopSignal", "motorSelectiveStop"]:
        col = f"{task}_stop_success_rate"
        if col not in qc_df.columns:
            continue

        # Filter to subject rows only (exclude mean/std)
        mask = ~qc_df.index.isin(["mean", "std"])
        data = qc_df.loc[mask, col].dropna()

        # Check lower bound
        low = data[data < STOP_SUCCESS_MIN]
        for subj, val in low.items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": "stop_success_rate",
                "metric_value": val,
                "threshold": f"< {STOP_SUCCESS_MIN}",
            })

        # Check upper bound
        high = data[data > STOP_SUCCESS_MAX]
        for subj, val in high.items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": "stop_success_rate",
                "metric_value": val,
                "threshold": f"> {STOP_SUCCESS_MAX}",
            })

    return pd.DataFrame(exclusions)


def check_stop_signal_go_accuracy(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check stop signal go accuracy is within acceptable bounds."""
    exclusions = []

    for task in ["stopSignal", "motorSelectiveStop"]:
        col = f"{task}_go_accuracy"
        if col not in qc_df.columns:
            continue

        mask = ~qc_df.index.isin(["mean", "std"])
        data = qc_df.loc[mask, col].dropna()

        for subj, val in data[data < STOP_SIGNAL_GO_ACC].items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": "go_accuracy",
                "metric_value": val,
                "threshold": f"< {STOP_SIGNAL_GO_ACC}",
            })

    return pd.DataFrame(exclusions)

def check_stop_signal_go_rt(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check stop signal go RT is within acceptable bounds."""
    exclusions = []

    for task in ["stopSignal", "motorSelectiveStop"]:
        col = f"{task}_go_rt"
        if col not in qc_df.columns:
            continue

        mask = ~qc_df.index.isin(["mean", "std"])
        data = qc_df.loc[mask, col].dropna()

        for subj, val in data[data > STOP_SIGNAL_GO_RT].items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": "go_rt",
                "metric_value": val,
                "threshold": f"> {STOP_SIGNAL_GO_RT}",
            })
            
    return pd.DataFrame(exclusions)

def check_motor_stop_noncrit_omission(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check motor selective stop noncrit signal omission rate."""
    col = "motorSelectiveStop_noncrit_signal_omission"
    if col not in qc_df.columns:
        return pd.DataFrame()

    mask = ~qc_df.index.isin(["mean", "std"])
    data = qc_df.loc[mask, col].dropna()

    high = data[data > MOTOR_STOP_NONCRIT_OMISSION_MAX]

    exclusions = [
        {
            "subject_id": subj,
            "task": "motorSelectiveStop",
            "metric": "noncrit_signal_omission",
            "metric_value": val,
            "threshold": f"> {MOTOR_STOP_NONCRIT_OMISSION_MAX}",
        }
        for subj, val in high.items()
    ]

    return pd.DataFrame(exclusions)


def check_discount_choice_pattern(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check for extreme choice patterns in discount task."""
    col = "discountFix_larger_later_pct"
    omission_col = "discountFix_omission_rate"

    if col not in qc_df.columns:
        return pd.DataFrame()

    mask = ~qc_df.index.isin(["mean", "std"])
    data = qc_df.loc[mask, [col, omission_col]].dropna()

    exclusions = []
    for subj in data.index:
        ll_pct = data.loc[subj, col]
        om_rate = data.loc[subj, omission_col] if omission_col in data.columns else 0

        # Check if only chose one option (accounting for omissions)
        total_choice_pct = ll_pct + om_rate

        if ll_pct == DISCOUNT_PROPORTION_MAX or total_choice_pct == DISCOUNT_PROPORTION_MAX:
            exclusions.append({
                "subject_id": subj,
                "task": "discountFix",
                "metric": "larger_later_pct",
                "metric_value": ll_pct,
                "threshold": f"= {DISCOUNT_PROPORTION_MAX} (only larger_later)",
            })
        elif ll_pct == DISCOUNT_PROPORTION_MIN or total_choice_pct == DISCOUNT_PROPORTION_MIN:
            exclusions.append({
                "subject_id": subj,
                "task": "discountFix",
                "metric": "larger_later_pct",
                "metric_value": ll_pct,
                "threshold": f"= {DISCOUNT_PROPORTION_MIN} (only smaller_sooner)",
            })

    return pd.DataFrame(exclusions)


def check_omission_rate(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check omission rate across all tasks."""
    exclusions = []

    for task in TASKS:
        col = f"{task}_omission_rate"
        if col not in qc_df.columns:
            continue

        mask = ~qc_df.index.isin(["mean", "std"])
        data = qc_df.loc[mask, col].dropna()

        high = data[data > OMISSION_RATE_MAX]
        for subj, val in high.items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": "omission_rate",
                "metric_value": val,
                "threshold": f"> {OMISSION_RATE_MAX}",
            })

    return pd.DataFrame(exclusions)


def check_truncation_rate(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check truncation rate across all tasks."""
    exclusions = []

    for task in TASKS:
        col = f"{task}_truncation_rate"
        if col not in qc_df.columns:
            continue

        mask = ~qc_df.index.isin(["mean", "std"])
        data = qc_df.loc[mask, col].dropna()

        high = data[data > TRUNCATION_RATE_MAX]
        for subj, val in high.items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": "truncation_rate",
                "metric_value": val,
                "threshold": f"> {TRUNCATION_RATE_MAX}",
            })

    return pd.DataFrame(exclusions)


def check_missing_data(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check for missing data (NaN values in key metrics)."""
    exclusions = []

    # Key metrics to check per task
    key_metrics = {
        "stopSignal": ["stop_success_rate", "SSRT"],
        "motorSelectiveStop": ["stop_success_rate", "SSRT"],
        "discountFix": ["larger_later_pct", "hyp_discount_rate_glm"],
        "manipulationTask": ["future_valence_avg", "present_valence_avg"],
    }

    mask = ~qc_df.index.isin(["mean", "std"])

    for task, metrics in key_metrics.items():
        for metric in metrics:
            col = f"{task}_{metric}"
            if col not in qc_df.columns:
                continue

            missing = qc_df.loc[mask, col].isna()
            for subj in missing[missing].index:
                exclusions.append({
                    "subject_id": subj,
                    "task": task,
                    "metric": metric,
                    "metric_value": np.nan,
                    "threshold": "missing",
                })

    return pd.DataFrame(exclusions)


def check_manip_pre_rating(subjects: list, data_path: str = SUBJECT_DATA_PATH) -> pd.DataFrame:
    """Check if subjects have pre-rating data for manipulation task."""
    exclusions = []

    for subj in subjects:
        subj_str = str(subj).lstrip("s")
        task_path = os.path.join(data_path, subj_str, "task")

        if not os.path.exists(task_path):
            has_prerating = False
        else:
            has_prerating = (
                os.path.exists(os.path.join(task_path, f"{subj_str}_preRating.csv")) or
                os.path.exists(os.path.join(task_path, f"{subj_str}_prerating.csv"))
            )

        if not has_prerating:
            exclusions.append({
                "subject_id": subj,
                "task": "manipulationTask",
                "metric": "pre_rating_file",
                "metric_value": 0,
                "threshold": "missing",
            })

    return pd.DataFrame(exclusions)


def check_minimum_valid_tasks(exclusion_df: pd.DataFrame, all_subjects: list) -> pd.DataFrame:
    """Flag subjects with too many task exclusions."""
    if exclusion_df.empty:
        return pd.DataFrame()

    # Count excluded tasks per subject
    task_counts = exclusion_df.groupby("subject_id")["task"].nunique()

    # Subjects with too few valid tasks
    max_excluded = len(TASKS) - MIN_VALID_TASKS
    flagged = task_counts[task_counts > max_excluded]

    exclusions = []
    for subj, n_excluded in flagged.items():
        # Add exclusion for any remaining tasks
        already_excluded = set(exclusion_df[exclusion_df.subject_id == subj]["task"])
        remaining = set(TASKS) - already_excluded

        for task in remaining:
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": "valid_task_count",
                "metric_value": len(TASKS) - n_excluded,
                "threshold": f"< {MIN_VALID_TASKS}",
            })

    return pd.DataFrame(exclusions)


def get_subjective_exclusions() -> pd.DataFrame:
    """Convert subjective exclusions to DataFrame format."""
    return pd.DataFrame([
        {
            "subject_id": exc["subject_id"],
            "task": exc["task"],
            "metric": "subjective_rating",
            "metric_value": np.nan,
            "threshold": exc["reason"],
        }
        for exc in SUBJECTIVE_EXCLUSIONS
    ])


def run_all_exclusion_checks(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Run all exclusion checks and return consolidated exclusion DataFrame.

    Returns DataFrame with columns: subject_id, task, metric, metric_value, threshold
    """
    # Get subject list (excluding summary rows)
    subjects = [idx for idx in qc_df.index if idx not in ("mean", "std")]

    # Run all checks
    checks = [
        get_subjective_exclusions(),
        check_stop_success_rate(qc_df),
        check_motor_stop_noncrit_omission(qc_df),
        check_discount_choice_pattern(qc_df),
        check_omission_rate(qc_df),
        check_truncation_rate(qc_df),
        check_missing_data(qc_df),
        check_manip_pre_rating(subjects),
    ]

    # Combine all exclusions
    exclusions = pd.concat([df for df in checks if not df.empty], ignore_index=True)

    # Check minimum valid tasks (needs existing exclusions)
    min_task_exclusions = check_minimum_valid_tasks(exclusions, subjects)
    if not min_task_exclusions.empty:
        exclusions = pd.concat([exclusions, min_task_exclusions], ignore_index=True)

    # Remove duplicates (same subject-task-metric)
    if not exclusions.empty:
        exclusions = exclusions.drop_duplicates(subset=["subject_id", "task", "metric"])

    return exclusions


def summarize_exclusions(exclusion_df: pd.DataFrame) -> dict:
    """Generate summary statistics from exclusion DataFrame."""
    if exclusion_df.empty:
        return {"total_exclusions": 0}

    return {
        "total_exclusions": len(exclusion_df),
        "unique_subjects_with_exclusions": exclusion_df.subject_id.nunique(),
        "exclusions_by_task": exclusion_df.groupby("task").size().to_dict(),
        "exclusions_by_metric": exclusion_df.groupby("metric").size().to_dict(),
    }
