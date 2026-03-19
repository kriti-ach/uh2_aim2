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


def _pick_dvars_column(columns: list[str]) -> str | None:
    """Choose available DVARS-like column with preference for raw dvars."""
    if "dvars" in columns:
        return "dvars"
    if "std_dvars" in columns:
        return "std_dvars"
    return None


def _pct_above_threshold(series: pd.Series, threshold: float) -> tuple[float, int, int]:
    """
    Compute percent of TRs above threshold.

    Returns:
        percent, n_above, n_valid
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
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

        fd = pd.to_numeric(df["framewise_displacement"], errors="coerce")
        fd_mean = float(fd.dropna().mean()) if fd.notna().any() else np.nan
        if np.isnan(fd_mean) or fd_mean < fd_mean_include_threshold_mm:
            continue

        fd_pct, fd_n_above, fd_n_valid = _pct_above_threshold(fd, fd_tr_threshold_mm)

        dvars_col = _pick_dvars_column(df.columns.tolist())
        if dvars_col is not None:
            dvars_pct, dvars_n_above, dvars_n_valid = _pct_above_threshold(
                df[dvars_col],
                dvars_tr_threshold,
            )
        else:
            dvars_pct, dvars_n_above, dvars_n_valid = np.nan, 0, 0

        row = {
            "subject_id": f"sub-{subject.lstrip('0') or '0'}",
            "task": task,
            "session": session,
            "run": int(run) if run else np.nan,
            "fd_mean_mm": fd_mean,
            "fd_n_trs_above_threshold": fd_n_above,
            "fd_n_trs_valid": fd_n_valid,
            "fd_percent_trs_above_threshold": fd_pct,
            "gt20pct_trs_fd_gt_0p5mm": bool(
                not np.isnan(fd_pct) and fd_pct > high_motion_percent_threshold
            ),
            "dvars_column_used": dvars_col,
            "dvars_n_trs_above_threshold": dvars_n_above,
            "dvars_n_trs_valid": dvars_n_valid,
            "dvars_percent_trs_above_threshold": dvars_pct,
            "gt20pct_trs_dvars_gt_1p5": bool(
                not np.isnan(dvars_pct) and dvars_pct > high_motion_percent_threshold
            ),
            "flagged_any_threshold": bool(
                (not np.isnan(fd_pct) and fd_pct > high_motion_percent_threshold)
                or (not np.isnan(dvars_pct) and dvars_pct > high_motion_percent_threshold)
            ),
            "confounds_tsv_path": str(confounds_path),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "task",
                "session",
                "run",
                "fd_mean_mm",
                "fd_n_trs_above_threshold",
                "fd_n_trs_valid",
                "fd_percent_trs_above_threshold",
                "gt20pct_trs_fd_gt_0p5mm",
                "dvars_column_used",
                "dvars_n_trs_above_threshold",
                "dvars_n_trs_valid",
                "dvars_percent_trs_above_threshold",
                "gt20pct_trs_dvars_gt_1p5",
                "flagged_any_threshold",
                "confounds_tsv_path",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["subject_id", "task", "session", "run"])
