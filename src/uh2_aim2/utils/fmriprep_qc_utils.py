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


def _combined_motion_spike_pct(
    fd_series: pd.Series,
    std_dvars_series: pd.Series,
    fd_threshold: float,
    std_dvars_threshold: float,
) -> tuple[float, int, int]:
    """
    Compute percent of TRs that are motion spikes by combined criterion:
    (FD > fd_threshold) OR (std_dvars > std_dvars_threshold).
    """
    fd_numeric = pd.Series(pd.to_numeric(fd_series, errors="coerce"))
    dvars_numeric = pd.Series(pd.to_numeric(std_dvars_series, errors="coerce"))

    valid_mask = fd_numeric.notna() | dvars_numeric.notna()
    spike_mask = (
        (fd_numeric > fd_threshold).fillna(False)
        | (dvars_numeric > std_dvars_threshold).fillna(False)
    )

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return np.nan, 0, 0

    n_spikes = int((spike_mask & valid_mask).sum())
    pct_spikes = (n_spikes / n_valid) * 100.0
    return pct_spikes, n_spikes, n_valid


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
        if "std_dvars" not in df.columns:
            continue
        std_dvars = pd.Series(df["std_dvars"])
        dvars_pct, dvars_n_above, dvars_n_valid = _pct_above_threshold(std_dvars, dvars_tr_threshold)
        spike_pct, spike_n, spike_valid = _combined_motion_spike_pct(
            fd_series=fd,
            std_dvars_series=std_dvars,
            fd_threshold=fd_tr_threshold_mm,
            std_dvars_threshold=dvars_tr_threshold,
        )

        row = {
            "subject_id": f"sub-{subject.lstrip('0') or '0'}",
            "task": task,
            "session": session,
            "run": int(run) if run is not None else np.nan,
            "fd_mean_mm": fd_mean,
            "fd_trs_above_threshold_count": fd_n_above,
            "fd_trs_valid_count": fd_n_valid,
            "fd_trs_above_threshold_percent": fd_pct,
            "fd_over_20pct_trs_above_threshold": bool(
                not np.isnan(fd_pct) and fd_pct > high_motion_percent_threshold
            ),
            "std_dvars_trs_above_threshold_count": dvars_n_above,
            "std_dvars_trs_valid_count": dvars_n_valid,
            "std_dvars_trs_above_threshold_percent": dvars_pct,
            "std_dvars_over_20pct_trs_above_threshold": bool(
                not np.isnan(dvars_pct) and dvars_pct > high_motion_percent_threshold
            ),
            "motion_spike_trs_count": spike_n,
            "motion_spike_trs_valid_count": spike_valid,
            "motion_spike_trs_percent": spike_pct,
            "motion_spikes_over_20pct": bool(
                not np.isnan(spike_pct) and spike_pct >= high_motion_percent_threshold
            ),
            "either_threshold_flagged": bool(
                not np.isnan(spike_pct) and spike_pct >= high_motion_percent_threshold
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
                "motion_spike_trs_count",
                "motion_spike_trs_valid_count",
                "motion_spike_trs_percent",
                "motion_spikes_over_20pct",
                "either_threshold_flagged",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["subject_id", "task"])
