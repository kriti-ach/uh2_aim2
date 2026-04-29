"""Exclusion detection for behavioral QC using vectorized pandas operations.

Outputs exclusions in format: subject_id, task, metric, metric_value, threshold
"""

import os

import numpy as np
import pandas as pd

from uh2_aim2.config import (
    BEHAVIOR_DATA_PROCESSED,
    BIDS_EVENT_FILES_TO_TRIM,
    MAX_LARGER_LATER_PROPORTION,
    MIN_LARGER_LATER_PROPORTION,
    MOTOR_STOP_NONCRIT_OMISSION_MAX,
    OMISSION_RATE_MAX,
    STOP_SUCCESS_MAX,
    STOP_SUCCESS_MIN,
    TASKS,
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

    task_to_column = {
        "stopSignal": "stopSignal_go_acc",
        "motorSelectiveStop": "motorSelectiveStop_crit_go_acc",
    }

    for task, col in task_to_column.items():
        if col not in qc_df.columns:
            continue

        mask = ~qc_df.index.isin(["mean", "std"])
        data = qc_df.loc[mask, col].dropna()

        for subj, val in data[data < STOP_SIGNAL_GO_ACC].items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": col.split(f"{task}_", 1)[1],
                "metric_value": val,
                "threshold": f"< {STOP_SIGNAL_GO_ACC}",
            })

    return pd.DataFrame(exclusions)

def check_stop_signal_go_rt(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Check stop signal go RT is within acceptable bounds."""
    exclusions = []

    task_to_column = {
        "stopSignal": "stopSignal_go_rt",
        "motorSelectiveStop": "motorSelectiveStop_crit_go_rt",
    }

    for task, col in task_to_column.items():
        if col not in qc_df.columns:
            continue

        mask = ~qc_df.index.isin(["mean", "std"])
        data = qc_df.loc[mask, col].dropna()

        for subj, val in data[data > STOP_SIGNAL_GO_RT].items():
            exclusions.append({
                "subject_id": subj,
                "task": task,
                "metric": col.split(f"{task}_", 1)[1],
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
    col = "discountFix_larger_later_proportion"
    omission_col = "discountFix_omission_rate"

    if col not in qc_df.columns:
        return pd.DataFrame()

    mask = ~qc_df.index.isin(["mean", "std"])
    data = qc_df.loc[mask, [col, omission_col]].dropna()

    exclusions = []
    for subj in data.index:
        larger_later_proportion = data.loc[subj, col]
        omission_rate = data.loc[subj, omission_col] if omission_col in data.columns else 0

        # Check if only chose one option (accounting for omissions)
        total_choice_proportion = larger_later_proportion + omission_rate

        if larger_later_proportion == MAX_LARGER_LATER_PROPORTION or total_choice_proportion == MAX_LARGER_LATER_PROPORTION:
            exclusions.append({
                "subject_id": subj,
                "task": "discountFix",
                "metric": "larger_later_proportion",
                "metric_value": larger_later_proportion,
                "threshold": f"= {MAX_LARGER_LATER_PROPORTION} (only larger_later_proportion)",
            })
        elif larger_later_proportion == MIN_LARGER_LATER_PROPORTION or total_choice_proportion == MIN_LARGER_LATER_PROPORTION:
            exclusions.append({
                "subject_id": subj,
                "task": "discountFix",
                "metric": "larger_later_proportion",
                "metric_value": larger_later_proportion,
                "threshold": f"= {MIN_LARGER_LATER_PROPORTION} (only smaller_sooner_proportion)",
            })

    return pd.DataFrame(exclusions)


def check_discount_missing_r_value(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Exclude discountFix rows where r2_value is missing, with stored reason."""
    r_col = "discountFix_r2_value"
    reason_col = "discountFix_r_value_reason"

    if r_col not in qc_df.columns:
        return pd.DataFrame()

    mask = ~qc_df.index.isin(["mean", "std"])
    data = qc_df.loc[mask]
    missing_r = data[data[r_col].isna()]

    exclusions = []
    for subj in missing_r.index:
        reason = ""
        if reason_col in data.columns and pd.notna(data.loc[subj, reason_col]):
            reason = str(data.loc[subj, reason_col])
        else:
            reason = "no reason captured"

        exclusions.append(
            {
                "subject_id": subj,
                "task": "discountFix",
                "metric": "r2_value",
                "metric_value": np.nan,
                "threshold": f"No r value because {reason}",
            }
        )

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


def _bids_trim_subject_task_pairs() -> set[tuple[int, str]]:
    """(subject_id, task) pairs configured for BIDS ``*_events.tsv`` trimming."""
    pairs: set[tuple[int, str]] = set()
    for entry in BIDS_EVENT_FILES_TO_TRIM:
        raw = entry["subject_id"]
        sid_str = str(raw).replace("sub-", "").replace("s", "")
        pairs.add((int(sid_str), str(entry["task"])))
    return pairs


def drop_omission_exclusions_for_bids_trim_targets(exclusion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove ``omission_rate`` exclusion rows when that subject/task is listed in
    ``BIDS_EVENT_FILES_TO_TRIM`` (event file will be trimmed instead).
    """
    if exclusion_df.empty or not BIDS_EVENT_FILES_TO_TRIM:
        return exclusion_df

    trim_pairs = _bids_trim_subject_task_pairs()

    def _in_trim(row: pd.Series) -> bool:
        sid = pd.to_numeric(row["subject_id"], errors="coerce")
        if pd.isna(sid):
            return False
        try:
            return (int(sid), str(row["task"])) in trim_pairs
        except (TypeError, ValueError):
            return False

    in_trim = exclusion_df.apply(_in_trim, axis=1)
    drop = exclusion_df["metric"].eq("omission_rate") & in_trim
    return exclusion_df.loc[~drop].reset_index(drop=True)


def check_missing_data(qc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing data (NaN values in key metrics).
    
    Returns DataFrame with columns: subject_id, task
    """
    missing_data = []

    # Key metrics to check per task
    key_metrics = {
        "stopSignal": ["stop_success_rate", "SSRT"],
        "motorSelectiveStop": ["stop_success_rate", "SSRT"],
        "discountFix": ["larger_later_proportion", "hyp_discount_rate_glm"],
        "manipulationTask": ["future_valence_avg", "present_valence_avg"],
    }

    mask = ~qc_df.index.isin(["mean", "std"])

    for task, metrics in key_metrics.items():
        # Check if ANY of the key metrics for this task are missing
        task_cols = [f"{task}_{metric}" for metric in metrics if f"{task}_{metric}" in qc_df.columns]
        
        if not task_cols:
            continue
        
        # Find subjects with missing data in any key metric for this task
        for subj in qc_df.loc[mask].index:
            if any(pd.isna(qc_df.loc[subj, col]) for col in task_cols):
                missing_data.append({
                    "subject_id": subj,
                    "task": task,
                })

    # Remove duplicates (in case multiple metrics were missing for same subject/task)
    if missing_data:
        return pd.DataFrame(missing_data).drop_duplicates()
    
    return pd.DataFrame(columns=["subject_id", "task"])


def check_manip_pre_rating(
    subjects: list, processed_path: str = BEHAVIOR_DATA_PROCESSED
) -> pd.DataFrame:
    """Check if subjects have pre-rating cleaned CSV in ``processed`` (e.g. ``1046_preRating_cleaned.csv``)."""
    exclusions = []

    for subj in subjects:
        subj_str = str(subj).lstrip("s")
        has_prerating = (
            os.path.isfile(os.path.join(processed_path, f"{subj_str}_preRating_cleaned.csv"))
            or os.path.isfile(os.path.join(processed_path, f"{subj_str}_prerating_cleaned.csv"))
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


def run_all_exclusion_checks(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Run all exclusion checks and return consolidated exclusion DataFrame.

    Returns DataFrame with columns: subject_id, task, metric, metric_value, threshold
    """
    # Get subject list (excluding summary rows)
    subjects = [idx for idx in qc_df.index if idx not in ("mean", "std")]

    # Run all checks
    checks = [
        check_stop_success_rate(qc_df),
        check_stop_signal_go_accuracy(qc_df),
        check_stop_signal_go_rt(qc_df),
        check_motor_stop_noncrit_omission(qc_df),
        check_discount_missing_r_value(qc_df),
        check_omission_rate(qc_df),
        check_manip_pre_rating(subjects),
    ]

    # Combine all exclusions
    exclusions = pd.concat([df for df in checks if not df.empty], ignore_index=True)

    # Remove only exact duplicate rows so multiple criteria for the same
    # subject/task are preserved in the exclusions output.
    if not exclusions.empty:
        exclusions = exclusions.drop_duplicates()
        exclusions = exclusions.copy()
        exclusions = drop_omission_exclusions_for_bids_trim_targets(exclusions)
        if exclusions.empty:
            exclusions["failed_multiple_criteria"] = pd.Series(dtype=bool)
            exclusions["other_criteria_failed"] = pd.Series(dtype=str)
            return exclusions

        exclusions["criterion_label"] = (
            exclusions["metric"].astype(str) + " (" + exclusions["threshold"].astype(str) + ")"
        )

        criteria_by_group = (
            exclusions.groupby(["subject_id", "task"])["criterion_label"]
            .apply(list)
            .to_dict()
        )
        exclusions["failed_multiple_criteria"] = (
            exclusions.groupby(["subject_id", "task"])["criterion_label"]
            .transform("size")
            .gt(1)
        )

        exclusions["other_criteria_failed"] = exclusions.apply(
            lambda row: ", ".join(
                criterion
                for criterion in criteria_by_group[(row["subject_id"], row["task"])]
                if criterion != row["criterion_label"]
            ),
            axis=1,
        )
        exclusions = exclusions.drop(columns=["criterion_label"])
    else:
        exclusions["failed_multiple_criteria"] = pd.Series(dtype=bool)
        exclusions["other_criteria_failed"] = pd.Series(dtype=str)

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
