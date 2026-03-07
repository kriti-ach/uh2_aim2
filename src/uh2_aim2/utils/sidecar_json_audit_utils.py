"""Utilities to audit BIDS sidecar JSON differences for flagged subjects."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUBJECT_PATTERN = re.compile(r"sub-([A-Za-z0-9]+)")
TASK_PATTERN = re.compile(r"task-([A-Za-z0-9]+)")
RUN_PATTERN = re.compile(r"run-([0-9]+)")
SESSION_PATTERN = re.compile(r"ses-([A-Za-z0-9]+)")


def _extract_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _flatten_json(
    obj: dict[str, Any],
    prefix: str = "",
    out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten nested JSON dict using dot-separated keys."""
    if out is None:
        out = {}

    for key, value in obj.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_json(value, prefix=flat_key, out=out)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[flat_key] = value
        elif isinstance(value, list):
            # Keep list as compact JSON string for categorical comparison
            out[flat_key] = json.dumps(value, separators=(",", ":"), sort_keys=False)
        else:
            out[flat_key] = str(value)
    return out


def _normalize_subject(subject: str) -> int | str:
    """Convert subject labels like '0479' or 's479' to int when possible."""
    subj = subject.replace("s", "").lstrip("0")
    if subj == "":
        subj = "0"
    try:
        return int(subj)
    except ValueError:
        return subject


def load_sidecar_json_table(bids_path: str) -> pd.DataFrame:
    """Load all BIDS sidecar JSON files into a flat dataframe."""
    rows: list[dict[str, Any]] = []
    bids_root = Path(bids_path)

    for json_file in sorted(bids_root.rglob("*_bold.json")):
        file_str = str(json_file)
        subject_raw = _extract_match(SUBJECT_PATTERN, file_str)
        task = _extract_match(TASK_PATTERN, json_file.name)
        run = _extract_match(RUN_PATTERN, json_file.name)
        session = _extract_match(SESSION_PATTERN, file_str)

        if subject_raw is None or task is None:
            continue

        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        flat = _flatten_json(payload)
        row = {
            "subject_id": _normalize_subject(subject_raw),
            "task": task,
            "session": session,
            "run": int(run) if run is not None else np.nan,
            "json_path": file_str,
        }
        row.update(flat)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["subject_id", "task", "session", "run", "json_path"])
    return pd.DataFrame(rows)


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit_flagged_subjects_vs_controls(
    sidecar_df: pd.DataFrame,
    flagged_subjects: list[int | str],
    z_threshold: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare flagged subjects against controls per task and return:
    - anomalies_df: row-level anomalies for flagged subjects
    - summary_df: aggregated key-level differences by task
    """
    if sidecar_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    flagged_set = set(flagged_subjects)
    base_cols = {"subject_id", "task", "session", "run", "json_path"}
    feature_cols = [col for col in sidecar_df.columns if col not in base_cols]

    anomalies: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for task, task_df in sidecar_df.groupby("task", dropna=False):
        controls = task_df[~task_df["subject_id"].isin(flagged_set)]
        flagged = task_df[task_df["subject_id"].isin(flagged_set)]
        if controls.empty or flagged.empty:
            continue

        for key in feature_cols:
            control_vals = controls[key].dropna()
            flagged_vals = flagged[key].dropna()
            if control_vals.empty or flagged_vals.empty:
                continue

            control_num = control_vals.apply(_safe_float)
            flagged_num = flagged_vals.apply(_safe_float)
            is_numeric = control_num.notna().mean() > 0.95 and flagged_num.notna().mean() > 0.95

            if is_numeric:
                control_num_vals = control_num.dropna().astype(float)
                flagged_num_vals = flagged_num.dropna().astype(float)
                if control_num_vals.empty or flagged_num_vals.empty:
                    continue

                c_mean = float(control_num_vals.mean())
                c_std = float(control_num_vals.std(ddof=0))
                f_mean = float(flagged_num_vals.mean())

                if c_std > 0:
                    z = (f_mean - c_mean) / c_std
                    different = abs(z) >= z_threshold
                    threshold_desc = f"|z| >= {z_threshold}"
                    metric_desc = f"z={z:.3f}"
                else:
                    z = np.nan
                    different = abs(f_mean - c_mean) > 0
                    threshold_desc = "control std = 0; value differs"
                    metric_desc = f"delta={f_mean - c_mean:.6g}"

                summary_rows.append(
                    {
                        "task": task,
                        "json_key": key,
                        "data_type": "numeric",
                        "different": bool(different),
                        "control_summary": f"mean={c_mean:.6g}, std={c_std:.6g}",
                        "flagged_summary": f"mean={f_mean:.6g}, n={len(flagged_num_vals)}",
                        "comparison_metric": metric_desc,
                        "comparison_threshold": threshold_desc,
                    }
                )

                if different:
                    for _, row in flagged.iterrows():
                        value = _safe_float(row.get(key))
                        if value is None:
                            continue
                        if c_std > 0:
                            per_row_z = (value - c_mean) / c_std
                            if abs(per_row_z) < z_threshold:
                                continue
                            reason = (
                                f"{key}={value:.6g} deviates from control mean {c_mean:.6g} "
                                f"(std={c_std:.6g}, z={per_row_z:.3f})"
                            )
                        else:
                            if value == c_mean:
                                continue
                            reason = (
                                f"{key}={value:.6g} differs from constant control value {c_mean:.6g}"
                            )
                        anomalies.append(
                            {
                                "subject_id": row["subject_id"],
                                "task": row["task"],
                                "session": row["session"],
                                "run": row["run"],
                                "json_key": key,
                                "flagged_value": value,
                                "control_reference": f"mean={c_mean:.6g}, std={c_std:.6g}",
                                "difference_reason": reason,
                                "json_path": row["json_path"],
                            }
                        )
            else:
                control_mode = control_vals.astype(str).mode()
                control_mode_val = control_mode.iloc[0] if not control_mode.empty else None
                flagged_unique = sorted(set(flagged_vals.astype(str)))
                different = any(val != control_mode_val for val in flagged_unique)

                summary_rows.append(
                    {
                        "task": task,
                        "json_key": key,
                        "data_type": "categorical",
                        "different": bool(different),
                        "control_summary": f"mode={control_mode_val}",
                        "flagged_summary": f"unique={flagged_unique}",
                        "comparison_metric": "value mismatch vs control mode",
                        "comparison_threshold": "flagged value != control mode",
                    }
                )

                if different:
                    for _, row in flagged.iterrows():
                        value = row.get(key)
                        if pd.isna(value):
                            continue
                        value_str = str(value)
                        if value_str == control_mode_val:
                            continue
                        anomalies.append(
                            {
                                "subject_id": row["subject_id"],
                                "task": row["task"],
                                "session": row["session"],
                                "run": row["run"],
                                "json_key": key,
                                "flagged_value": value_str,
                                "control_reference": f"mode={control_mode_val}",
                                "difference_reason": (
                                    f"{key}={value_str} differs from control mode {control_mode_val}"
                                ),
                                "json_path": row["json_path"],
                            }
                        )

    anomalies_df = pd.DataFrame(anomalies)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["different", "task", "json_key"], ascending=[False, True, True])
    if not anomalies_df.empty:
        anomalies_df = anomalies_df.sort_values(["subject_id", "task", "json_key", "run"])
    return anomalies_df, summary_df
