"""Utilities to audit Flywheel pfile metadata without downloading files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from uh2_aim2.config import TASKS


RUN_RE = re.compile(r"run[-_]?([0-9]+)", re.IGNORECASE)


def normalize_subject_id(subject: int | str) -> str:
    """Normalize subject to numeric string used in Flywheel labels."""
    token = str(subject).strip().replace("sub-", "").replace("s", "")
    token = token.lstrip("0") or "0"
    return token


def infer_task_from_scan_name(scan_name: str) -> str | None:
    """Infer task from acquisition label/scan name using lightweight matching."""
    lowered = scan_name.lower()
    task_aliases = {
        "discountFix": ["discountfix", "discount_fix", "task-discountfix"],
        "motorSelectiveStop": ["motorselectivestop", "motor_selective_stop", "task-motorselectivestop"],
        "stopSignal": ["stopsignal", "stop_signal", "task-stopsignal"],
        "manipulationTask": ["manipulationtask", "manipulation_task", "task-manipulationtask"],
        "rest": ["rest", "task-rest"],
    }
    for task, aliases in task_aliases.items():
        if any(alias in lowered for alias in aliases):
            return task
    return None


def extract_run_number(scan_name: str) -> int | None:
    """Parse run number from scan name."""
    match = RUN_RE.search(scan_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _search_nested_for_key(obj: Any, key_name: str) -> Any:
    """Recursively find a key in nested dict/list metadata structures."""
    if isinstance(obj, dict):
        if key_name in obj:
            return obj[key_name]
        for value in obj.values():
            found = _search_nested_for_key(value, key_name)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _search_nested_for_key(item, key_name)
            if found is not None:
                return found
    return None


def extract_gain_fields(file_info: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    """Extract aps_r1, aps_r2, aps_tg from Flywheel file info recursively."""
    if not isinstance(file_info, dict):
        return None, None, None

    aps_r1 = _search_nested_for_key(file_info, "aps_r1")
    aps_r2 = _search_nested_for_key(file_info, "aps_r2")
    aps_tg = _search_nested_for_key(file_info, "aps_tg")
    return aps_r1, aps_r2, aps_tg


def _iter_pfile_records(acquisition: Any, subject_label: str) -> list[dict[str, Any]]:
    """Build row records for pfile-type files in one Flywheel acquisition."""
    records: list[dict[str, Any]] = []
    acq_label = str(getattr(acquisition, "label", ""))
    acq_id = str(getattr(acquisition, "id", ""))
    task = infer_task_from_scan_name(acq_label)
    run = extract_run_number(acq_label)

    files = getattr(acquisition, "files", []) or []
    for fw_file in files:
        file_type = str(getattr(fw_file, "type", "") or "").lower()
        file_name = str(getattr(fw_file, "name", ""))
        # Flywheel may mark these as pfile or with pfile-like naming
        is_pfile = (file_type == "pfile") or file_name.endswith(".7") or ("pfile" in file_name.lower())
        if not is_pfile:
            continue

        file_info = getattr(fw_file, "info", None)
        aps_r1, aps_r2, aps_tg = extract_gain_fields(file_info)
        records.append(
            {
                "subject_id": subject_label,
                "acquisition_id": acq_id,
                "scan_name": acq_label,
                "task": task,
                "task_found_in_scan_name": task is not None,
                "run": run,
                "file_name": file_name,
                "file_type": file_type,
                "aps_r1": aps_r1,
                "aps_r2": aps_r2,
                "aps_tg": aps_tg,
                "metadata_present": any(x is not None for x in (aps_r1, aps_r2, aps_tg)),
            }
        )
    return records


def collect_flagged_subject_pfile_metadata(
    fw_client: Any,
    project_path: str,
    flagged_subjects: list[int | str],
) -> pd.DataFrame:
    """
    Collect pfile metadata for flagged subjects via Flywheel SDK (metadata only).

    This does not download pfiles; it reads acquisition/file metadata only.
    """
    project = fw_client.lookup(project_path)
    requested = {normalize_subject_id(s) for s in flagged_subjects}

    rows: list[dict[str, Any]] = []
    subjects = project.subjects()  # SDK query; lightweight relative to file download
    for subject in subjects:
        subj_label_raw = str(getattr(subject, "label", ""))
        subj_norm = normalize_subject_id(subj_label_raw)
        if subj_norm not in requested:
            continue

        for session in subject.sessions():
            acquisitions = session.acquisitions()
            for acquisition in acquisitions:
                rows.extend(_iter_pfile_records(acquisition, f"sub-{subj_norm}"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "acquisition_id",
                "scan_name",
                "task",
                "task_found_in_scan_name",
                "run",
                "file_name",
                "file_type",
                "aps_r1",
                "aps_r2",
                "aps_tg",
                "metadata_present",
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["subject_id", "task", "run", "scan_name"]).reset_index(drop=True)


def summarize_gain_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize within-subject task/run gain variability."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "task",
                "n_scans",
                "aps_r1_unique",
                "aps_r2_unique",
                "aps_tg_unique",
                "aps_r1_values",
                "aps_r2_values",
                "aps_tg_values",
                "possible_gain_instability",
            ]
        )

    def _unique_sorted(values: pd.Series) -> list[Any]:
        cleaned = [v for v in values.dropna().tolist()]
        try:
            return sorted(set(cleaned))
        except TypeError:
            return list(dict.fromkeys(cleaned))

    grouped = (
        df.groupby(["subject_id", "task"], dropna=False)
        .agg(
            n_scans=("scan_name", "count"),
            aps_r1_values=("aps_r1", _unique_sorted),
            aps_r2_values=("aps_r2", _unique_sorted),
            aps_tg_values=("aps_tg", _unique_sorted),
        )
        .reset_index()
    )
    grouped["aps_r1_unique"] = grouped["aps_r1_values"].apply(len)
    grouped["aps_r2_unique"] = grouped["aps_r2_values"].apply(len)
    grouped["aps_tg_unique"] = grouped["aps_tg_values"].apply(len)
    grouped["possible_gain_instability"] = (
        (grouped["aps_r1_unique"] > 1) | (grouped["aps_r2_unique"] > 1) | (grouped["aps_tg_unique"] > 1)
    )
    return grouped


def save_pfile_audit_outputs(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str,
) -> tuple[Path, Path]:
    """Save detailed and summary CSV outputs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = out_dir / "flywheel_pfile_metadata_detailed.csv"
    summary_path = out_dir / "flywheel_pfile_metadata_summary.csv"
    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return detailed_path, summary_path
