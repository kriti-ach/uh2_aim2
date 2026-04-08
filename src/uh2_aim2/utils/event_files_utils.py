"""Utilities for trimming BIDS behavioral event TSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from uh2_aim2.config import (
    BIDS_EVENT_FILES_TO_TRIM,
    BIDS_PATH,
    NO_RESPONSE,
    TRIMMED_EVENT_OUTPUT_BIDS_DIR,
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


def _is_bids_match(file_path: Path, target: TrimTarget) -> bool:
    file_name = file_path.name
    subject = target.subject_id
    task = target.task
    subject_match = f"sub-{subject}" in str(file_path)
    task_match = f"task-{task}" in file_name
    return subject_match and task_match and file_name.endswith("_events.tsv")


def _find_matching_bids_files(
    root_dir: Path,
    targets: list[TrimTarget],
) -> list[tuple[Path, TrimTarget]]:
    matches: list[tuple[Path, TrimTarget]] = []
    for file_path in sorted(root_dir.rglob("*_events.tsv")):
        for target in targets:
            if _is_bids_match(file_path, target):
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
    Trim configured files under ``BIDS_PATH`` only.

    Always writes preview outputs to ``trimmed_event_file_outputs/bids_outputs``.
    If apply_to_source is True, also overwrites the original BIDS files.
    """
    targets = _load_targets(BIDS_EVENT_FILES_TO_TRIM)
    bids_root = Path(BIDS_PATH)
    preview_root = Path(TRIMMED_EVENT_OUTPUT_ROOT)
    preview_bids_root = Path(TRIMMED_EVENT_OUTPUT_BIDS_DIR)

    preview_root.mkdir(parents=True, exist_ok=True)
    preview_bids_root.mkdir(parents=True, exist_ok=True)

    matches = _find_matching_bids_files(bids_root, targets)

    records: list[dict[str, str | int | bool | None]] = []

    for source_file, target in matches:
        df = pd.read_csv(source_file, sep="\t")
        trimmed_df, trim_idx = _trim_dataframe(df)

        preview_path = _preview_output_path(source_file, bids_root, preview_bids_root)

        _write_trimmed(trimmed_df, preview_path)

        if apply_to_source:
            _write_trimmed(trimmed_df, source_file)

        records.append(
            {
                "source": "bids",
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
