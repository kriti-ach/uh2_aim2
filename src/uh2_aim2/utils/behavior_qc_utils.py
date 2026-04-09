"""Utility functions for behavioral QC calculations using vectorized pandas operations."""

import os
from math import ceil, floor

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from uh2_aim2.config import (
    BEHAVIOR_SAMPLE_DATA,
    CONDITIONS,
    CONDITION_COLUMN,
    EXPERIMENTOR_WAIT_TRIAL_ID,
    GO_TRIAL_TYPES,
    MAX_GO_RT,
    MAX_SSD,
    MIN_SSD,
    NO_RESPONSE,
    STOP_TRIAL_TYPES,
    SECONDS_TO_MILLISECONDS,
)
# =============================================================================
# CORE METRIC CALCULATIONS
# =============================================================================


def calc_omission_rate(df: pd.DataFrame, task: str) -> float:
    """Calculate omission (non-response) rate for a task."""
    if df.empty:
        return 1.0

    task_map = {
        "stopSignal": lambda d: d[d.trial_type == "go"],
        "motorSelectiveStop": lambda d: d[d.trial_type.isin(GO_TRIAL_TYPES["motorSelectiveStop"])],
        "manipulationTask": lambda d: d[(d.trial_id == "current_rating") & (d.trial_type != "no_stim")],
    }

    trials = task_map.get(task, lambda d: d)(df)
    if len(trials) == 0:
        return np.nan

    return (trials.key_press == NO_RESPONSE).mean()

def calc_commission_rate(df: pd.DataFrame, task: str) -> dict[str, float]:
    """Calculate commission rate for stop tasks."""
    if df.empty:
        return {}

    if task == "stopSignal":
        go_trials = df[df.trial_type == "go"]
        if len(go_trials) == 0:
            return {"commission_rate": np.nan}
        return {
            "commission_rate": ((go_trials.key_press != go_trials.correct_response) & (go_trials.key_press != NO_RESPONSE)).mean()
        }
    
    elif task == "motorSelectiveStop":
        # For motor selective stop, return dict with different commission rates
        crit_go_trials = df[df.trial_type == "crit_go"]
        crit_commission = (
            (crit_go_trials.key_press != crit_go_trials.correct_response).mean()
            if len(crit_go_trials) > 0 else np.nan
        )
        
        noncrit_signal_trials = df[df.trial_type == "noncrit_signal"]
        noncrit_signal_commission = (
            (noncrit_signal_trials.key_press != noncrit_signal_trials.correct_response).mean()
            if len(noncrit_signal_trials) > 0 else np.nan
        )
        
        noncrit_nosignal_trials = df[df.trial_type == "noncrit_nosignal"]
        noncrit_nosignal_commission = (
            (noncrit_nosignal_trials.key_press != noncrit_nosignal_trials.correct_response).mean()
            if len(noncrit_nosignal_trials) > 0 else np.nan
        )
        
        return {
            "crit_commission_rate": crit_commission,
            "noncrit_signal_commission_rate": noncrit_signal_commission,
            "noncrit_nosignal_commission_rate": noncrit_nosignal_commission,
        }
    
    return {}

def calc_ssrt(df: pd.DataFrame, task: str) -> float:
    """Calculate Stop Signal Reaction Time using integration method."""
    if task not in STOP_TRIAL_TYPES:
        return np.nan

    stop_types = STOP_TRIAL_TYPES[task]
    go_type = "go" if task == "stopSignal" else "crit_go"

    go_trials = df[df.trial_type == go_type].copy()
    stop_trials = df[df.trial_type.isin([stop_types["success"], stop_types["failure"]])]

    if len(go_trials) == 0 or len(stop_trials) == 0:
        return np.nan

    # Replace missing RTs with max
    go_trials.loc[go_trials.response_time * SECONDS_TO_MILLISECONDS == NO_RESPONSE, "response_time"] =  MAX_GO_RT 
    sorted_go = go_trials.response_time.sort_values()

    prob_stop_failure = 1 - stop_trials.stopped.mean()
    nth = prob_stop_failure * (len(sorted_go) - 1)
    nth_rt = sorted_go.iloc[[floor(nth), ceil(nth)]].mean()

    return (nth_rt - stop_trials.SS_delay.mean()) * SECONDS_TO_MILLISECONDS


def calc_stop_success_rate(df: pd.DataFrame, task: str) -> float:
    """Calculate stop success rate for stop tasks."""
    if task not in STOP_TRIAL_TYPES:
        return np.nan

    stop_types = STOP_TRIAL_TYPES[task]
    stop_trials = df[df.trial_type.isin([stop_types["success"], stop_types["failure"]])]

    if len(stop_trials) == 0:
        return np.nan

    return (stop_trials.trial_type == stop_types["success"]).mean()


def calc_discount_rate_glm(
    df: pd.DataFrame,
    subject_id: str | int | None = None,
) -> tuple[float, float, str | None]:
    """Calculate hyperbolic discount rate using GLM."""
    data = df.copy()
    subj_label = f"subject={subject_id}" if subject_id is not None else "subject=unknown"

    data["patient"] = np.where(
        data.choice == "larger_later", 1,
        np.where(data.choice == "smaller_sooner", 0, np.nan)
    )
    data = data.dropna(subset=["patient"])

    if len(data) == 0:
        return np.nan, np.nan, f"{subj_label}: no valid patient choices"

    data["indiff_k"] = (
        (data.large_amount.astype(float) - data.small_amount.astype(float)) /
        (data.small_amount.astype(float) * data.later_delay.astype(float))
    )
    
    data = data[np.isfinite(data.indiff_k)]
    
    if len(data) == 0:
        return np.nan, np.nan, f"{subj_label}: no finite indiff_k values"

    unique_choices = set(data.patient)
    if unique_choices == {0.0}:
        return data.indiff_k.max(), np.nan, f"{subj_label}: all choices were smaller_sooner"
    if unique_choices == {1.0}:
        return data.indiff_k.min(), np.nan, f"{subj_label}: all choices were larger_later"

    if data["indiff_k"].std() < 1e-10:
        return np.nan, np.nan, f"{subj_label}: near-zero variance in indiff_k"

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='Inverting hessian failed')
            
            model = smf.glm(
                "patient ~ indiff_k", 
                data=data, 
                family=sm.families.Binomial()
            ).fit(maxiter=100, disp=False)
        
        if 'Intercept' not in model.params or 'indiff_k' not in model.params:
            reason = (
                f"{subj_label}: missing GLM params"
            )
            return data.indiff_k.median(), np.nan, reason
            
        intercept = model.params['Intercept']
        slope = model.params['indiff_k']
        
        if not np.isfinite(intercept) or not np.isfinite(slope):
            reason = (
                f"{subj_label}: non-finite GLM params"
            )
            return data.indiff_k.median(), np.nan, reason
        
        if abs(slope) < 1e-10:
            reason = (
                f"{subj_label}: slope near zero"
            )
            return data.indiff_k.median(), np.nan, reason
            
        k = -intercept / slope
        
        if k < 0 or not np.isfinite(k) or k > 1000:
            reason = (
                f"{subj_label}: invalid k={k}"
            )
            return data.indiff_k.median(), np.nan, reason
        
        # Calculate r2
        if hasattr(model, 'llf') and hasattr(model, 'llnull'):
            if np.isfinite(model.llf) and np.isfinite(model.llnull) and model.llnull != 0:
                r2 = 1 - (model.llf / model.llnull)
                if np.isfinite(r2) and 0 <= r2 <= 1:
                    return k, r2, None
        
        reason = (
            f"{subj_label}: R value calculation failed"
        )
        return k, np.nan, reason
        
    except Exception as e:
        reason = (
            f"{subj_label}: GLM exception {type(e).__name__}"
        )
        return data.indiff_k.median(), np.nan, reason


# =============================================================================
# QC SUMMARY FUNCTIONS
# =============================================================================

def _empty_qc_subject_df() -> pd.DataFrame:
    """Per-subject QC table with no rows (after filter/load produced no trials)."""
    return pd.DataFrame(index=pd.Index([], name="subject_id"))


def standardize_subject_numbers(subj: str | int) -> int:
    """Standardize subject numbers to remove the "s" prefix."""
    if isinstance(subj, str) and subj.startswith("s"):
        return int(subj.replace("s", ""))
    return int(subj)


def _rows_for_subject_id(df: pd.DataFrame, subj_int: int) -> pd.DataFrame:
    """
    Rows for one participant given ``worker_id`` variants in cleaned files.

    Some files use ``s1021``, others plain ``1021``; string/int mismatch used to
    yield an empty subset and bogus metrics (e.g. omission_rate == 1.0).
    """
    w_raw = pd.Series(df["worker_id"], index=df.index)
    w = w_raw.astype(str).str.strip()
    token = str(subj_int)
    str_mask = w.isin({token, f"s{token}", f"sub-{token}"})
    num = pd.to_numeric(w_raw, errors="coerce")
    num_mask = num == float(subj_int)
    return df.loc[str_mask | num_mask]


def compute_qc_summary(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compute both RT and accuracy QC metrics for a task."""
    rt_df = compute_rt_summary(df, task)
    acc_df = compute_acc_summary(df, task)
    
    # Merge RT and accuracy metrics
    combined = pd.concat([rt_df, acc_df], axis=1)
    
    # Add mean and std rows
    combined.loc["mean"] = combined.mean(numeric_only=True)
    combined.loc["std"] = combined.std(numeric_only=True)
    
    return combined


def compute_rt_summary(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compute RT summary statistics by subject for a task."""
    if df.empty or "worker_id" not in df.columns:
        return _empty_qc_subject_df()

    conditions = CONDITIONS.get(task, [])
    if not conditions:
        conditions = df.trial_type.dropna().unique().tolist()
    cond_col = CONDITION_COLUMN.get(task, "trial_type")

    # Group by subject and condition, compute mean RT
    results = []
    for raw in df["worker_id"].dropna().unique():
        subj = standardize_subject_numbers(raw)
        subj_df = _rows_for_subject_id(df, subj)
        row = {"subject_id": subj}

        for cond in conditions:
            cond_rt = subj_df[subj_df[cond_col] == cond].response_time.mean() * SECONDS_TO_MILLISECONDS
            row[f"{cond}_rt"] = cond_rt

        # Add mean SSD for stop tasks
        if task in STOP_TRIAL_TYPES:
            row["mean_SSD"] = subj_df.SS_delay.mean() * SECONDS_TO_MILLISECONDS

        results.append(row)

    if not results:
        return _empty_qc_subject_df()
    return pd.DataFrame(results).set_index("subject_id")


def compute_acc_summary(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compute accuracy summary statistics by subject for a task."""
    # Use task-specific functions
    if task == "discountFix":
        return _compute_discount_acc(df)
    if task == "manipulationTask":
        return _compute_manip_acc(df)

    return _compute_standard_acc(df, task)


def _compute_standard_acc(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compute accuracy for standard tasks (stop signal, motor stop)."""
    if df.empty or "worker_id" not in df.columns:
        return _empty_qc_subject_df()

    conditions = CONDITIONS.get(task, [])
    cond_col = CONDITION_COLUMN.get(task, "trial_type")

    results = []
    for raw in df["worker_id"].dropna().unique():
        subj = standardize_subject_numbers(raw)
        subj_df = _rows_for_subject_id(df, subj)
        row = {"subject_id": subj}

        # Accuracy by condition
        for cond in conditions:
            cond_df = subj_df[subj_df[cond_col] == cond]
            row[f"{cond}_acc"] = cond_df.correct.mean() if len(cond_df) > 0 else np.nan

        # Stop task specific metrics
        if task in STOP_TRIAL_TYPES:
            stop_types = STOP_TRIAL_TYPES[task]
            stop_trials = subj_df[subj_df.trial_type.isin([stop_types["success"], stop_types["failure"]])]

            row["stop_success_rate"] = calc_stop_success_rate(subj_df, task)
            row["SSRT"] = calc_ssrt(subj_df, task)
            row["max_SSD_count"] = (stop_trials.SS_delay == MAX_SSD).sum() if len(stop_trials) > 0 else 0
            row["min_SSD_count"] = (stop_trials.SS_delay == MIN_SSD).sum() if len(stop_trials) > 0 else 0

            # Commission rates (works for both tasks)
            commission_rates = calc_commission_rate(subj_df, task)
            row.update(commission_rates)

            # Motor stop specific: omission by trial type
            if task == "motorSelectiveStop":
                for tt in ["noncrit_signal", "noncrit_nosignal", "crit_go"]:
                    tt_df = subj_df[subj_df.trial_type == tt]
                    row[f"{tt}_omission"] = (tt_df.key_press == NO_RESPONSE).mean() if len(tt_df) > 0 else np.nan

        if task == "stopSignal":
            row["omission_rate"] = calc_omission_rate(subj_df, task)
        results.append(row)

    if not results:
        return _empty_qc_subject_df()
    return pd.DataFrame(results).set_index("subject_id")


def _compute_discount_acc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy metrics for discount task."""
    if df.empty or "worker_id" not in df.columns:
        return _empty_qc_subject_df()

    results = []
    for raw in df["worker_id"].dropna().unique():
        subj = standardize_subject_numbers(raw)
        subj_df = _rows_for_subject_id(df, subj)

        larger_later_proportion = (subj_df.choice == "larger_later").mean()
        k, r2, r_value_reason = calc_discount_rate_glm(subj_df, subject_id=subj)

        results.append({
            "subject_id": subj,
            "larger_later_proportion": larger_later_proportion,
            "k_value": k,
            "r2_value": r2,
            "r_value_reason": r_value_reason,
            "omission_rate": calc_omission_rate(subj_df, "discountFix"),
        })

    if not results:
        return _empty_qc_subject_df()
    return pd.DataFrame(results).set_index("subject_id")


def _compute_manip_acc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy metrics for manipulation task."""
    if df.empty or "worker_id" not in df.columns:
        return _empty_qc_subject_df()

    trial_types = CONDITIONS["manipulationTask"]

    results = []
    for raw in df["worker_id"].dropna().unique():
        subj = standardize_subject_numbers(raw)
        subj_df = _rows_for_subject_id(df, subj)
        rating_df = subj_df[(subj_df.trial_id == "current_rating") & (subj_df.trial_type != "no_stim")]

        row = {"subject_id": subj}

        for tt in trial_types:
            tt_df = rating_df[rating_df.trial_type == tt]
            row[f"{tt}_avg"] = tt_df.response.mean() if len(tt_df) > 0 else np.nan

            # Response distribution
            if len(tt_df) > 0:
                for resp in range(1, 6):
                    row[f"{tt}_{resp}"] = (tt_df.response == resp).mean()

        row["omission_rate"] = calc_omission_rate(subj_df, "manipulationTask")
        results.append(row)

    if not results:
        return _empty_qc_subject_df()
    return pd.DataFrame(results).set_index("subject_id")


def final_sample_task_csv_path(subject: str | int, task: str) -> str:
    """Path to one task file under ``BEHAVIOR_SAMPLE_DATA`` (analysis sample, not QC input)."""
    sid = str(standardize_subject_numbers(subject))
    return os.path.join(BEHAVIOR_SAMPLE_DATA, sid, "task", f"{sid}_{task}.csv")


def remove_practice_stage_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict rows used for QC.

    If ``exp_stage`` exists, drop rows whose value normalizes to ``practice``
    (case-insensitive); all other rows are kept (including NaN / non-test labels).
    If ``exp_stage`` is absent and ``trial_id`` exists, keep rows strictly after
    the first ``trial_id == EXPERIMENTOR_WAIT_TRIAL_ID`` in file order. If that
    marker is absent, return the dataframe unchanged. If neither column exists,
    return unchanged.
    """
    if df.empty:
        return df
    if "exp_stage" in df.columns:
        stage = df["exp_stage"]
        is_practice = stage.notna() & (
            stage.astype(str).str.strip().str.lower() == "practice"
        )
        return df.loc[~is_practice].copy()
    if "trial_id" in df.columns:
        tid = df["trial_id"].astype(str).str.strip()
        wait = (tid == EXPERIMENTOR_WAIT_TRIAL_ID).to_numpy()
        if not wait.any():
            return df.copy()
        first_pos = int(np.flatnonzero(wait)[0])
        return df.iloc[first_pos + 1 :].copy()
    return df.copy()