"""Sanitize UH2 AIM2 BIDS ``*_events.tsv`` columns (task-specific + global drops)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

# Dropped from every task’s events file when present.
ALL_TASK_COLUMNS_TO_DROP = (
    "group_RT",
    "worker_id",
    "experiment_exp_id",
    "junk",
    "duration",
)

TASK_SPECIFIC_COLUMNS_TO_DROP: dict[str, tuple[str, ...]] = {
    "discountFix": ("trial_id", "inverse_delay", "subjective_choice_value"),
    "manipulationTask": ("junk_tmp",),
    "stopSignal": ("passed_check",),
    "motorSelectiveStop": ("passed_check",),
}

# If max numeric ``block_duration`` is below this, treat values as **seconds** and multiply by 1000;
# otherwise leave unchanged (assumed already milliseconds).
MANIPULATION_BLOCK_DURATION_MAX_FOR_SECONDS_HEURISTIC = 5000.0


def parse_task_from_events_filename(file_name: str) -> str | None:
    """Return BIDS task label (e.g. ``discountFix``) from ``*_events.tsv`` filename."""
    if not file_name.endswith("_events.tsv"):
        return None
    stem = file_name[: -len("_events.tsv")]
    for part in stem.split("_"):
        if part.startswith("task-"):
            return part[5:]
    return None


def _drop_columns_if_present(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    to_drop = [c for c in columns if c in df.columns]
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def sanitize_events_dataframe(df: pd.DataFrame, task: str | None) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply column removals and, for manipulationTask only, optionally convert
    ``block_duration`` from seconds to ms when values look like seconds.

    Returns ``(dataframe, manipulation_block_duration_notes)``.
    """
    notes: list[str] = []
    out = df.copy()

    if task == "manipulationTask" and "block_duration" in out.columns:
        num = pd.to_numeric(out["block_duration"], errors="coerce")
        finite = num.dropna()
        if finite.empty:
            notes.append("manipulationTask:block_duration_unchanged(no_finite_numeric_values)")
        elif float(finite.max()) < MANIPULATION_BLOCK_DURATION_MAX_FOR_SECONDS_HEURISTIC:
            out["block_duration"] = num * 1000.0
            notes.append(
                "manipulationTask:block_duration_seconds_to_ms"
                f"(max={float(finite.max()):.4g}<{MANIPULATION_BLOCK_DURATION_MAX_FOR_SECONDS_HEURISTIC})"
            )
        else:
            notes.append(
                "manipulationTask:block_duration_unchanged_assumed_ms"
                f"(max={float(finite.max()):.4g}>={MANIPULATION_BLOCK_DURATION_MAX_FOR_SECONDS_HEURISTIC})"
            )

    extra = TASK_SPECIFIC_COLUMNS_TO_DROP.get(task or "", ())
    out = _drop_columns_if_present(out, extra)
    out = _drop_columns_if_present(out, ALL_TASK_COLUMNS_TO_DROP)
    return out, notes


def iter_subject_func_events(bids_root: Path, subject_id: str) -> list[Path]:
    """All ``*_events.tsv`` under ``sub-<id>/func/``."""
    sid = str(subject_id).strip().replace("sub-", "").replace("s", "")
    sub_dir = bids_root / f"sub-{sid}" / "func"
    if not sub_dir.is_dir():
        return []
    return sorted(p for p in sub_dir.glob("*_events.tsv") if p.is_file())


def relative_path_under_bids(source: Path, bids_root: Path) -> Path:
    return source.resolve().relative_to(bids_root.resolve())


def backup_bids_events_tsv(
    source_tsv: Path,
    bids_root: Path,
    backup_root: Path,
    *,
    skip_if_backup_exists: bool = True,
) -> tuple[Path, bool]:
    """
    Copy ``source_tsv`` under ``backup_root`` preserving path relative to ``bids_root``.

    Returns ``(backup_path, copied)``. If ``skip_if_backup_exists`` and the backup file
    already exists, returns ``(path, False)`` without overwriting (keeps first snapshot).
    """
    rel = relative_path_under_bids(source_tsv, bids_root)
    dest = backup_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if skip_if_backup_exists and dest.exists():
        return dest, False
    shutil.copy2(source_tsv, dest)
    return dest, True


def sanitize_events_file(
    source_tsv: Path,
    bids_root: Path,
    output_root: Path,
) -> tuple[Path, str | None, list[str]]:
    """
    Read ``source_tsv``, sanitize, write under ``output_root`` preserving path
    relative to ``bids_root``.

    Returns ``(written_path, task, list of applied change notes)``.
    """
    task = parse_task_from_events_filename(source_tsv.name)
    df = pd.read_csv(source_tsv, sep="\t")
    before_cols = list(df.columns)

    out_df, man_notes = sanitize_events_dataframe(df, task)
    after_cols = list(out_df.columns)

    rel = relative_path_under_bids(source_tsv, bids_root)
    dest = output_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dest, sep="\t", index=False)

    dropped = sorted(set(before_cols) - set(after_cols))
    notes: list[str] = []
    notes.extend(man_notes)
    if dropped:
        notes.append(f"dropped_columns={dropped}")
    return dest, task, notes
