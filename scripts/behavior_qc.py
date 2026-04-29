#!/usr/bin/env python
"""
Behavioral QC Pipeline for UH2 AIM2.

Runs quality control on behavioral data and outputs:
- Per-task QC CSVs (containing both RT and accuracy metrics)
- Combined exclusion file (subject_id, task, metric, metric_value, threshold)
"""

import os
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from uh2_aim2.config import (
    BEHAVIOR_DATA_PROCESSED,
    BEHAVIOR_QC_PATH,
    BEHAVIOR_EXCLUSIONS_TIMING_ALLOWLIST,
    BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD,
    BEHAVIOR_TIMING_QC_CSV,
    BEHAVIOR_TIMING_QC_FLAGGED_CSV,
    EXCLUSION_REASON_EF_TIMING_OFF,
    EXCLUSION_REASON_MISSING_BEHAVIOR_FILE,
    FINAL_EXCLUSIONS_JSON_PATH,
    NO_RESPONSE,
    SECONDS_TO_MILLISECONDS,
    TASKS,
)

# jsPsych key codes -> Likert 1–5 in cleaned manipulationTask CSVs (thumb … pinky).
_MANIP_RATING_KEYCODE_TO_LIKERT = {66: 1, 89: 2, 71: 3, 82: 4, 77: 5}
from uh2_aim2.utils.behavior_exclusion_utils import (
    run_all_exclusion_checks,
    summarize_exclusions,
    check_missing_data,
)
from uh2_aim2.utils.behavior_flagging_utils import run_all_flagging_checks
from uh2_aim2.utils.behavior_plot_utils import plot_all_task_histograms
from uh2_aim2.utils.behavior_qc_utils import (
    compute_qc_summary,
    missing_raw_behavior_csv_pairs,
    remove_practice_stage_rows,
    standardize_subject_numbers,
)
from uh2_aim2.utils.behavior_timing_qc_utils import run_behavior_timing_qc


def _format_subject_id(subject_value: object) -> str:
    """Normalize subject values to BIDS-style ids, e.g. sub-1021."""
    raw = str(subject_value).strip()
    if raw.startswith("sub-"):
        token = raw.replace("sub-", "")
    elif raw.startswith("s"):
        token = raw[1:]
    else:
        token = raw
    token = token.lstrip("0") or "0"
    return f"sub-{token}"


_REASON_PRIORITY = {
    "behavioral exclusion": 0,
    EXCLUSION_REASON_EF_TIMING_OFF: 1,
    EXCLUSION_REASON_MISSING_BEHAVIOR_FILE: 2,
}


def _pair_in_behavior_exclusions_allowlist(subject_id: object, task: object) -> bool:
    sid_for_std = subject_id if isinstance(subject_id, (str, int)) else str(subject_id)
    s_int = standardize_subject_numbers(sid_for_std)
    t_str = str(task).strip()
    for sid_a, task_a in BEHAVIOR_EXCLUSIONS_TIMING_ALLOWLIST:
        if int(sid_a) == int(s_int) and str(task_a) == t_str:
            return True
    return False


def _timing_rows_for_exclusions_json(timing_df: pd.DataFrame) -> pd.DataFrame:
    """Same flag logic as ``qc_behavior_timings`` flagged CSV, plus unreadable CSV rows."""
    if timing_df.empty:
        return timing_df
    fr = timing_df["flag_reason"].astype(str)
    delta_ok = pd.to_numeric(timing_df["delta"], errors="coerce")
    thresh = float(BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD)
    mask = (~timing_df["ok"]) & (
        fr.eq("missing csv")
        | (delta_ok >= thresh)
        | fr.str.startswith("read error")
    )
    return timing_df.loc[mask]


def _merge_behavioral_exclusion_json_records(
    exclusion_df: pd.DataFrame,
    missing_raw_df: pd.DataFrame,
    timing_df: pd.DataFrame,
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}

    def upsert(subject_value: object, task_value: object, reason: str) -> None:
        key_sub = _format_subject_id(subject_value)
        task_s = str(task_value).strip()
        key = (key_sub, task_s)
        pri = _REASON_PRIORITY.get(reason, 99)
        if key not in merged or pri < merged[key][0]:
            merged[key] = (pri, {"subject": key_sub, "task": task_s, "reason": reason})

    if not exclusion_df.empty:
        pairs = exclusion_df[["subject_id", "task"]].drop_duplicates()
        for _, row in pairs.iterrows():
            upsert(row["subject_id"], row["task"], "behavioral exclusion")

    timing_sub = _timing_rows_for_exclusions_json(timing_df)
    if not timing_sub.empty:
        for _, row in timing_sub.iterrows():
            if _pair_in_behavior_exclusions_allowlist(row["subject_id"], row["task"]):
                continue
            fr = str(row["flag_reason"]).strip()
            reason = (
                EXCLUSION_REASON_MISSING_BEHAVIOR_FILE
                if fr == "missing csv"
                else EXCLUSION_REASON_EF_TIMING_OFF
            )
            upsert(row["subject_id"], row["task"], reason)

    if not missing_raw_df.empty:
        for _, row in missing_raw_df.iterrows():
            if _pair_in_behavior_exclusions_allowlist(row["subject_id"], row["task"]):
                continue
            upsert(row["subject_id"], row["task"], EXCLUSION_REASON_MISSING_BEHAVIOR_FILE)

    return [
        pair[1]
        for pair in sorted(
            merged.values(),
            key=lambda x: (x[1]["subject"], x[1]["task"]),
        )
    ]


def _update_final_exclusions_json(
    exclusion_df: pd.DataFrame,
    missing_raw_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    json_path: str,
) -> None:
    """
    Replace ``behavioral_exclusions`` in ``exclusions.json`` under ``BEHAVIOR_QC_PATH``.

    Combines metric exclusions, missing raw CSV pairs, and behavioral timing QC flags.
    Rows matching ``BEHAVIOR_EXCLUSIONS_TIMING_ALLOWLIST`` are omitted (timing + missing-file).

    Keeps all other top-level exclusion sections unchanged.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}

    payload.setdefault("behavioral_exclusions", [])
    payload.setdefault("fmriprep_exclusions", [])
    payload.setdefault("other_exclusions", [])

    payload["behavioral_exclusions"] = _merge_behavioral_exclusion_json_records(
        exclusion_df, missing_raw_df, timing_df
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Coerce mixed boolean/text/int columns to bool."""
    text = series.astype(str).str.strip().str.lower()
    true_set = {"true", "1", "yes", "y"}
    return text.isin(true_set)


def _harmonize_cleaned_task_df(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Harmonize cleaned behavioral CSVs to columns expected by QC utilities.

    After harmonization, restricts trials via :func:`remove_practice_stage_rows`
    (drops ``exp_stage == practice`` when that column exists; if it is absent,
    keeps rows after ``trial_id == 'experimentor_wait'``).
    """
    out = df.copy()

    # Standardize worker id strings for subject parsing.
    if "worker_id" in out.columns:
        out["worker_id"] = out["worker_id"].astype(str).str.strip()

    # Cleaned files store RT in milliseconds; QC utilities expect response_time in seconds.
    if "response_time" not in out.columns and "rt" in out.columns:
        rt_ms = pd.Series(pd.to_numeric(out["rt"], errors="coerce"), index=out.index)
        out["response_time"] = rt_ms / float(SECONDS_TO_MILLISECONDS)

    # Stop-signal delays are in milliseconds in cleaned files.
    if "SS_delay" in out.columns:
        ssd_ms = pd.Series(pd.to_numeric(out["SS_delay"], errors="coerce"), index=out.index)
        out["SS_delay"] = ssd_ms / float(SECONDS_TO_MILLISECONDS)

    if task == "stopSignal" and "SS_trial_type" in out.columns:
        ss_type = out["SS_trial_type"].astype(str).str.strip().str.lower()
        stopped = _to_bool_series(pd.Series(out["stopped"], index=out.index)) if "stopped" in out.columns else pd.Series(False, index=out.index)
        trial_type = pd.Series("", index=out.index, dtype=object)
        trial_type.loc[ss_type == "go"] = "go"
        trial_type.loc[(ss_type == "stop") & stopped] = "stop_success"
        trial_type.loc[(ss_type == "stop") & (~stopped)] = "stop_failure"
        out["trial_type"] = trial_type

    if task == "motorSelectiveStop" and {"condition", "SS_trial_type"}.issubset(out.columns):
        cond = out["condition"].astype(str).str.strip().str.lower()
        ss_type = out["SS_trial_type"].astype(str).str.strip().str.lower()
        stopped = _to_bool_series(pd.Series(out["stopped"], index=out.index)) if "stopped" in out.columns else pd.Series(False, index=out.index)

        trial_type = pd.Series("", index=out.index, dtype=object)
        trial_type.loc[(cond == "go") & (ss_type == "go")] = "crit_go"
        trial_type.loc[(cond == "ignore") & (ss_type == "go")] = "noncrit_nosignal"
        trial_type.loc[(cond == "ignore") & (ss_type == "stop")] = "noncrit_signal"
        trial_type.loc[(cond == "stop") & (ss_type == "stop") & stopped] = "crit_stop_success"
        trial_type.loc[(cond == "stop") & (ss_type == "stop") & (~stopped)] = "crit_stop_failure"
        # Fallback for unexpected combinations.
        trial_type.loc[(cond == "go") & (trial_type == "")] = "crit_go"
        out["trial_type"] = trial_type

    if task == "manipulationTask":
        if {"trial_id", "stim_type", "which_cue"}.issubset(out.columns):
            is_rating = out["trial_id"].astype(str).str.strip() == "current_rating"
            stim = out["stim_type"].astype(str).str.strip().str.lower()
            cue = out["which_cue"].astype(str).str.strip().str.upper()
            # Cleaned exports use ``food`` for smoking/valence images; some pipelines use ``smoking``.
            valence_stim = stim.isin(("smoking", "food"))

            qc_label = pd.Series(np.nan, index=out.index, dtype=object)
            qc_label.loc[is_rating & (cue == "LATER") & valence_stim] = "future_valence"
            qc_label.loc[is_rating & (cue == "LATER") & (stim == "neutral")] = "future_neutral"
            qc_label.loc[is_rating & (cue == "NOW") & valence_stim] = "present_valence"
            qc_label.loc[is_rating & (cue == "NOW") & (stim == "neutral")] = "present_neutral"
            qc_label.loc[is_rating & qc_label.isna()] = "no_stim"

            base_tt = (
                out["trial_type"].copy()
                if "trial_type" in out.columns
                else pd.Series("", index=out.index, dtype=object)
            )
            out["trial_type"] = base_tt
            out.loc[is_rating, "trial_type"] = qc_label.loc[is_rating]

            # Cleaned CSV stores key codes; BIDS/events often store 1–5. Map codes for QC stats only.
            if "response" in out.columns:
                rnum = pd.to_numeric(out["response"], errors="coerce")
                likert = pd.Series(np.nan, index=out.index, dtype=float)
                for key_code, scale_val in _MANIP_RATING_KEYCODE_TO_LIKERT.items():
                    likert = likert.where(rnum != key_code, float(scale_val))
                already_scale = rnum.notna() & rnum.ge(1) & rnum.le(5)
                likert = likert.where(~already_scale, rnum)
                likert = likert.where(rnum != NO_RESPONSE, np.nan)
                out.loc[is_rating, "response"] = likert.loc[is_rating]

    out = remove_practice_stage_rows(out)
    return out


def load_task_data(behavior_data_path: str = BEHAVIOR_DATA_PROCESSED) -> dict[str, pd.DataFrame]:
    """Load cleaned behavioral CSV files for all tasks."""
    data = {}

    for task in TASKS:
        files = glob(os.path.join(behavior_data_path, f"*_{task}_cleaned.csv"))
        if files:
            dfs = []
            for f in files:
                df = pd.read_csv(f)
                dfs.append(_harmonize_cleaned_task_df(df, task))
            data[task] = pd.concat(dfs, ignore_index=True)
            print(f"  {task}: {len(files)} files, {len(data[task])} trials")
            if data[task].empty:
                print(
                    f"  WARNING: {task}: no trials left after harmonize / stage filter "
                    "(check exp_stage vs practice, or rows after trial_id 'experimentor_wait')."
                )
        else:
            print(f"  {task}: no files found")

    return data


def run_qc_pipeline(
    behavior_data_path: str = BEHAVIOR_DATA_PROCESSED,
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
    print("\n[1/7] Loading data...")
    task_data = load_task_data(behavior_data_path)

    # Compute QC metrics (both RT and accuracy)
    print("\n[2/7] Computing QC metrics...")
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
    print("\n[3/7] Checking for missing data...")
    missing_df = check_missing_data(all_qc_data)

    # Run exclusion checks (WITHOUT missing data check)
    print("\n[4/7] Running exclusion checks...")
    exclusion_df = run_all_exclusion_checks(all_qc_data)

    # Run flagging checks
    print("\n[5/7] Running flagging checks...")
    flags_df = run_all_flagging_checks(all_qc_data)
    if not flags_df.empty:
        excluded_pairs = set(
            zip(exclusion_df["subject_id"], exclusion_df["task"])
        ) if not exclusion_df.empty else set()
        flags_df["excluded"] = [
            (subj, task) in excluded_pairs
            for subj, task in zip(flags_df["subject_id"], flags_df["task"])
        ]
    else:
        flags_df["excluded"] = pd.Series(dtype=bool)

    # Behavioral timing QC (raw CSVs; drives timing exclusions in exclusions.json)
    print("\n[6/7] Behavioral timing QC (raw)...")
    timing_df = run_behavior_timing_qc()
    timing_out_dir = os.path.dirname(os.path.abspath(BEHAVIOR_TIMING_QC_CSV))
    if timing_out_dir:
        os.makedirs(timing_out_dir, exist_ok=True)
    timing_df.to_csv(BEHAVIOR_TIMING_QC_CSV, index=False)
    delta_timing = pd.to_numeric(timing_df["delta"], errors="coerce")
    thresh = float(BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD)
    flagged_timing = timing_df[
        (~timing_df["ok"])
        & (
            timing_df["flag_reason"].astype(str).eq("missing csv")
            | (delta_timing >= thresh)
        )
    ].copy()
    flagged_timing.to_csv(BEHAVIOR_TIMING_QC_FLAGGED_CSV, index=False)
    print(f"  {BEHAVIOR_TIMING_QC_CSV}")
    print(f"  {BEHAVIOR_TIMING_QC_FLAGGED_CSV}")

    missing_raw_df = missing_raw_behavior_csv_pairs(behavior_data_path)

    # Generate histograms
    print("\n[7/7] Generating QC histograms...")
    plot_all_task_histograms(qc_results, output_path)

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
    _update_final_exclusions_json(
        exclusion_df, missing_raw_df, timing_df, FINAL_EXCLUSIONS_JSON_PATH
    )
    print(f"  {FINAL_EXCLUSIONS_JSON_PATH}")

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
