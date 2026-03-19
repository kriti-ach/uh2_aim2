"""Utilities for lightweight fMRIPrep motion QC from confounds TSV files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


SUBJECT_RE = re.compile(r"sub-([A-Za-z0-9]+)")
TASK_RE = re.compile(r"task-([A-Za-z0-9]+)")


def _extract(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _compute_motion_spikes(
    fd_series: pd.Series,
    std_dvars_series: pd.Series,
    fd_threshold: float,
    std_dvars_threshold: float,
) -> tuple[float, int, int]:
    """
    Compute percent of TRs that are motion spikes.
    A TR is a motion spike if: (FD > fd_threshold) OR (std_dvars > std_dvars_threshold).
    
    Returns:
        percent_spikes, n_spikes, n_valid_trs
    """
    fd_numeric = pd.to_numeric(fd_series, errors="coerce")
    dvars_numeric = pd.to_numeric(std_dvars_series, errors="coerce")
    
    # A TR is valid if we have at least one valid measurement
    valid_mask = fd_numeric.notna() | dvars_numeric.notna()
    n_valid = int(valid_mask.sum())
    
    if n_valid == 0:
        return np.nan, 0, 0
    
    # A TR is a spike if either threshold is exceeded
    spike_mask = (
        (fd_numeric > fd_threshold).fillna(False) | 
        (dvars_numeric > std_dvars_threshold).fillna(False)
    )
    
    n_spikes = int(spike_mask.sum())
    percent_spikes = (n_spikes / n_valid) * 100.0
    
    return percent_spikes, n_spikes, n_valid


def collect_fmriprep_motion_metrics(
    fmriprep_root: str,
    fd_threshold_mm: float = 0.5,
    std_dvars_threshold: float = 1.5,
    high_motion_threshold_percent: float = 20.0,
) -> pd.DataFrame:
    """
    Collect scan-level motion metrics from fMRIPrep confounds TSVs.
    
    For each scan, identifies TRs as motion spikes if either:
    - FD > fd_threshold_mm, OR
    - std_dvars > std_dvars_threshold
    
    Then flags scans where >= high_motion_threshold_percent of TRs are motion spikes.
    """
    root = Path(fmriprep_root)
    confounds_files = sorted(root.rglob("*_desc-confounds_timeseries.tsv"))

    rows: list[dict[str, object]] = []
    
    for confounds_path in confounds_files:
        file_name = confounds_path.name
        file_str = str(confounds_path)

        subject = _extract(SUBJECT_RE, file_name) or _extract(SUBJECT_RE, file_str)
        task = _extract(TASK_RE, file_name)
        
        if subject is None or task is None:
            continue

        df = pd.read_csv(confounds_path, sep="\t")
        
        # Check required columns exist
        if "framewise_displacement" not in df.columns or "std_dvars" not in df.columns:
            continue

        fd = df["framewise_displacement"]
        std_dvars = df["std_dvars"]
        
        # Compute mean FD
        fd_numeric = pd.to_numeric(fd, errors="coerce")
        fd_mean = float(fd_numeric.mean()) if fd_numeric.notna().any() else np.nan
        
        spike_pct, spike_n, n_valid = _compute_motion_spikes(
            fd_series=fd,
            std_dvars_series=std_dvars,
            fd_threshold=fd_threshold_mm,
            std_dvars_threshold=std_dvars_threshold,
        )

        row = {
            "subject_id": f"sub-{subject.lstrip('0') or '0'}",
            "task": task,
            "fd_mean_mm": fd_mean,
            "motion_spike_trs_count": spike_n,
            "total_trs": n_valid,
            "motion_spike_percent": spike_pct,
            "high_motion_flag": bool(
                not np.isnan(spike_pct) and spike_pct >= high_motion_threshold_percent
            ),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "task",
                "fd_mean_mm",
                "motion_spike_trs_count",
                "total_trs",
                "motion_spike_percent",
                "high_motion_flag",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["subject_id", "task"])