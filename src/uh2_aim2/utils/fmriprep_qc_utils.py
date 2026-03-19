"""Utilities for lightweight fMRIPrep motion QC from confounds TSV files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SUBJECT_RE = re.compile(r"sub-([A-Za-z0-9]+)")
TASK_RE = re.compile(r"task-([A-Za-z0-9]+)")
RUN_RE = re.compile(r"run-([0-9]+)")
SESSION_RE = re.compile(r"ses-([A-Za-z0-9]+)")


def _extract(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _pct_above_threshold(series: pd.Series, threshold: float) -> tuple[float, int, int]:
    """
    Compute percent of TRs above threshold.

    Returns:
        percent, n_above, n_valid
    """
    numeric = pd.Series(pd.to_numeric(series, errors="coerce")).dropna()
    n_valid = int(len(numeric))
    if n_valid == 0:
        return np.nan, 0, 0
    n_above = int((numeric > threshold).sum())
    percent = (n_above / n_valid) * 100.0
    return percent, n_above, n_valid


def collect_fmriprep_motion_metrics(
    fmriprep_root: str,
    fd_tr_threshold_mm: float,
    dvars_tr_threshold: float,
    high_motion_percent_threshold: float,
    fd_mean_include_threshold_mm: float,
) -> pd.DataFrame:
    """
    Collect scan-level motion metrics from fMRIPrep confounds TSVs.

    Keeps scans with mean FD >= fd_mean_include_threshold_mm.
    """
    root = Path(fmriprep_root)
    confounds_files = sorted(root.rglob("*_desc-confounds_timeseries.tsv"))

    rows: list[dict[str, object]] = []
    for confounds_path in confounds_files:
        file_name = confounds_path.name
        file_str = str(confounds_path)

        subject = _extract(SUBJECT_RE, file_name) or _extract(SUBJECT_RE, file_str)
        task = _extract(TASK_RE, file_name)
        run = _extract(RUN_RE, file_name)
        session = _extract(SESSION_RE, file_str)
        if subject is None or task is None:
            continue

        df = pd.read_csv(confounds_path, sep="\t")
        if "framewise_displacement" not in df.columns:
            continue

        fd = pd.Series(pd.to_numeric(df["framewise_displacement"], errors="coerce"))
        fd_mean = float(fd.dropna().mean()) if fd.notna().any() else np.nan
        if np.isnan(fd_mean) or fd_mean < fd_mean_include_threshold_mm:
            continue

        fd_pct, fd_n_above, fd_n_valid = _pct_above_threshold(fd, fd_tr_threshold_mm)

        # Use standardized DVARS only (requested)
        if "std_dvars" in df.columns:
            std_dvars = pd.Series(df["std_dvars"])
            dvars_pct, dvars_n_above, dvars_n_valid = _pct_above_threshold(std_dvars, dvars_tr_threshold)
        else:
            dvars_pct, dvars_n_above, dvars_n_valid = np.nan, 0, 0

        row = {
            "subject_id": f"sub-{subject.lstrip('0') or '0'}",
            "task": task,
            "fd_mean_mm": fd_mean,
            f"fd_trs_above_{fd_tr_threshold_mm}mm_count": fd_n_above,
            "fd_trs_valid_count": fd_n_valid,
            "fd_trs_above_threshold_percent": fd_pct,
            f"fd_over_{high_motion_percent_threshold}pct_trs_above_threshold": bool(
                not np.isnan(fd_pct) and fd_pct > high_motion_percent_threshold
            ),
            f"std_dvars_trs_above_{dvars_tr_threshold}count": dvars_n_above,
            "std_dvars_trs_valid_count": dvars_n_valid,
            "std_dvars_trs_above_threshold_percent": dvars_pct,
            f"std_dvars_over_{high_motion_percent_threshold}pct_trs_above_threshold": bool(
                not np.isnan(dvars_pct) and dvars_pct > high_motion_percent_threshold
            ),
            "either_threshold_flagged": bool(
                (not np.isnan(fd_pct) and fd_pct > high_motion_percent_threshold)
                or (not np.isnan(dvars_pct) and dvars_pct > high_motion_percent_threshold)
            ),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "task",
                "fd_mean_mm",
                "fd_trs_above_threshold_count",
                "fd_trs_valid_count",
                "fd_trs_above_threshold_percent",
                "fd_over_20pct_trs_above_threshold",
                "std_dvars_trs_above_threshold_count",
                "std_dvars_trs_valid_count",
                "std_dvars_trs_above_threshold_percent",
                "std_dvars_over_20pct_trs_above_threshold",
                "either_threshold_flagged",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["subject_id", "task"])
