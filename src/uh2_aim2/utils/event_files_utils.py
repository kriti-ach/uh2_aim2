"""Utilities for trimming behavioral event files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BIDS_EVENT_FILES_TO_TRIM,
    BIDS_PATH,
    EVENT_FILES_PATH,
    EVENT_FILES_TO_TRIM,
    NO_RESPONSE,
    TRIMMED_EVENT_OUTPUT_BIDS_DIR,
    TRIMMED_EVENT_OUTPUT_EVENT_FILES_DIR,
    TRIMMED_EVENT_OUTPUT_ROOT,
)


@dataclass(frozen=True)
class TrimTarget:
    subject_id: str
    task: str


def _normalize_subject(subject_id: str | int) -> str:
    """Normalize subject id to numeric string (s1021/sub-1021 -> 1021)."""
    return str(subject_id).replace("sub-", "").replace("s", "")


def _load_targets(raw_targets: list[dict[str, str | int]]) -> list[TrimTarget]:
    """Convert config target dicts into typed trim targets."""
    return [
        TrimTarget(subject_id=_normalize_subject(target["subject_id"]), task=str(target["task"]))
        for target in raw_targets
    ]


def _is_match(file_path: Path, target: TrimTarget, source: str) -> bool:
    """Check if a file belongs to the configured subject/task target."""
    file_name = file_path.name
    subject = target.subject_id
    task = target.task

    if source == "event_files":
        subject_match = (f"s{subject}" in file_name) or (subject in file_name)
        task_match = task in file_name
        return subject_match and task_match and file_name.endswith("_events.tsv")

    subject_match = f"sub-{subject}" in str(file_path)
    task_match = f"task-{task}" in file_name
    return subject_match and task_match and file_name.endswith("_events.tsv")


def _find_matching_files(
    root_dir: Path,
    targets: list[TrimTarget],
    source: str,
) -> list[tuple[Path, TrimTarget]]:
    """Find all event files in root_dir matching configured trim targets."""
    matches: list[tuple[Path, TrimTarget]] = []
    for file_path in sorted(root_dir.rglob("*_events.tsv")):
        for target in targets:
            if _is_match(file_path, target, source):
                matches.append((file_path, target))
                break
    return matches


def _find_trim_index(key_press: pd.Series) -> int | None:
    """
    Find first index where key_press == -1 and all remaining key_press are -1.
    """
    numeric = pd.Series(pd.to_numeric(key_press, errors="coerce"), index=key_press.index)
    if numeric.isna().all():
        return None

    no_response = numeric.to_numpy() == NO_RESPONSE
    if not no_response.any():
        return None

    suffix_all = np.logical_and.accumulate(no_response[::-1])[::-1]
    candidate_indices = np.flatnonzero(no_response & suffix_all)
    if candidate_indices.size == 0:
        return None

    return int(candidate_indices[0])


def _trim_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    """Trim dataframe based on configured key_press stopping rule."""
    if "key_press" not in df.columns:
        return df.copy(), None

    key_press_series = pd.Series(df.loc[:, "key_press"])
    trim_idx = _find_trim_index(key_press_series)
    if trim_idx is None:
        return df.copy(), None

    return df.iloc[:trim_idx].copy(), trim_idx


def _preview_output_path(source_file: Path, source_root: Path, preview_root: Path) -> Path:
    """Build preview output path preserving relative structure."""
    return preview_root / source_file.relative_to(source_root)


def _write_trimmed(trimmed_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed_df.to_csv(output_path, sep="\t", index=False)


def trim_configured_event_files(apply_to_source: bool = False) -> pd.DataFrame:
    """
    Trim configured files in both event_files and BIDS trees.

    Always writes preview outputs to `trimmed_event_file_outputs`.
    If apply_to_source is True, also overwrites the original source files.
    """
    event_targets = _load_targets(EVENT_FILES_TO_TRIM)
    bids_targets = _load_targets(BIDS_EVENT_FILES_TO_TRIM)

    event_root = Path(EVENT_FILES_PATH)
    bids_root = Path(BIDS_PATH)
    preview_root = Path(TRIMMED_EVENT_OUTPUT_ROOT)
    preview_event_root = Path(TRIMMED_EVENT_OUTPUT_EVENT_FILES_DIR)
    preview_bids_root = Path(TRIMMED_EVENT_OUTPUT_BIDS_DIR)

    preview_root.mkdir(parents=True, exist_ok=True)
    preview_event_root.mkdir(parents=True, exist_ok=True)
    preview_bids_root.mkdir(parents=True, exist_ok=True)

    event_matches = _find_matching_files(event_root, event_targets, source="event_files")
    bids_matches = _find_matching_files(bids_root, bids_targets, source="bids")

    all_matches: list[tuple[str, Path, TrimTarget]] = [
        ("event_files", source_file, target) for source_file, target in event_matches
    ] + [
        ("bids", source_file, target) for source_file, target in bids_matches
    ]

    records: list[dict[str, str | int | bool | None]] = []

    for source, source_file, target in all_matches:
        df = pd.read_csv(source_file, sep="\t")
        trimmed_df, trim_idx = _trim_dataframe(df)

        if source == "event_files":
            preview_path = _preview_output_path(source_file, event_root, preview_event_root)
        else:
            preview_path = _preview_output_path(source_file, bids_root, preview_bids_root)

        _write_trimmed(trimmed_df, preview_path)

        if apply_to_source:
            _write_trimmed(trimmed_df, source_file)

        records.append(
            {
                "source": source,
                "subject_id": target.subject_id,
                "task": target.task,
                "source_file": str(source_file),
                "preview_file": str(preview_path),
                "original_rows": len(df),
                "trimmed_rows": len(trimmed_df),
                "rows_removed": len(df) - len(trimmed_df),
                "trim_applied": trim_idx is not None,
                "trim_index": trim_idx,
                "applied_to_source": apply_to_source,
            }
        )

    summary_df = pd.DataFrame(records)
    summary_path = preview_root / "trim_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df
