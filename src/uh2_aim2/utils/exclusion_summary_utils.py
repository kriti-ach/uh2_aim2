"""
Summarize sample sizes before vs after applying ``exclusions.json``.

Total subjects counted = union of subjects under ``BEHAVIOR_DATA_RAW``, ``BIDS_PATH/sub-*``, and any
subject referenced in the JSON. Initial “has task” uses raw behavioral CSVs for the
four behavioral tasks and a BIDS scan for ``rest`` (``*task-rest*_bold.nii*`` under
``sub-<id>/``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from uh2_aim2.config import (
    BEHAVIOR_DATA_RAW,
    BIDS_PATH,
    FULL_QC_CANONICAL_TASKS,
)
from uh2_aim2.utils.exclusion_json_utils import canonical_task_key, normalize_subject_bids


def load_exclusions_payload(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def collect_excluded_subject_task_pairs(
    payload: dict,
    canonical_tasks: tuple[str, ...] = FULL_QC_CANONICAL_TASKS,
) -> set[tuple[str, str]]:
    """``(sub-XXX, canonical_task)`` for every JSON row whose task matches ``canonical_tasks``."""
    out: set[tuple[str, str]] = set()
    for key in ("behavioral_exclusions", "fmriprep_exclusions", "other_exclusions"):
        for row in payload.get(key, []):
            sub = normalize_subject_bids(row.get("subject", ""))
            ct = canonical_task_key(str(row.get("task", "")), canonical_tasks)
            if ct is not None:
                out.add((sub, ct))
    return out


def _subjects_from_behavior_raw(behavior_raw: str) -> set[str]:
    if not os.path.isdir(behavior_raw):
        return set()
    out: set[str] = set()
    for entry in os.listdir(behavior_raw):
        subdir = os.path.join(behavior_raw, entry)
        if os.path.isdir(subdir):
            out.add(normalize_subject_bids(entry))
    return out


def _subjects_from_bids(bids_path: str) -> set[str]:
    root = Path(bids_path)
    if not root.is_dir():
        return set()
    return {
        normalize_subject_bids(p.name)
        for p in root.glob("sub-*")
        if p.is_dir()
    }


def _subjects_from_exclusions_payload(payload: dict) -> set[str]:
    out: set[str] = set()
    for key in ("behavioral_exclusions", "fmriprep_exclusions", "other_exclusions"):
        for row in payload.get(key, []):
            out.add(normalize_subject_bids(row.get("subject", "")))
    return out


def _subject_has_behavioral_raw(
    behavior_raw: str, sub_bids: str, task: str
) -> bool:
    token = sub_bids.replace("sub-", "")
    path = os.path.join(behavior_raw, token, f"{token}_{task}.csv")
    return os.path.isfile(path)


def _subject_has_rest_bids(bids_path: str, sub_bids: str) -> bool:
    if not sub_bids.startswith("sub-"):
        sub_bids = normalize_subject_bids(sub_bids)
    sub_dir = Path(bids_path) / sub_bids
    if not sub_dir.is_dir():
        return False
    for pat in ("*task-rest*_bold.nii.gz", "*task-rest*_bold.nii"):
        for _ in sub_dir.rglob(pat):
            return True
    return False


def build_subject_universe(
    payload: dict,
    behavior_raw: str,
    bids_path: str,
) -> set[str]:
    return (
        _subjects_from_behavior_raw(behavior_raw)
        | _subjects_from_bids(bids_path)
        | _subjects_from_exclusions_payload(payload)
    )


def compute_exclusion_sample_summary(
    exclusions_json_path: str,
    *,
    behavior_raw: str | None = None,
    bids_path: str | None = None,
    canonical_tasks: tuple[str, ...] = FULL_QC_CANONICAL_TASKS,
) -> dict[str, object]:
    """
    Returns counts for reporting; keys include ``total_subjects_n``, ``initial_complete_n``,
    ``final_complete_n``, ``per_task`` (list of dicts).
    """
    behavior_raw = behavior_raw if behavior_raw is not None else BEHAVIOR_DATA_RAW
    bids_path = bids_path if bids_path is not None else BIDS_PATH

    payload = load_exclusions_payload(exclusions_json_path)
    excluded_pairs = collect_excluded_subject_task_pairs(payload, canonical_tasks)
    all_subjects = build_subject_universe(payload, behavior_raw, bids_path)

    behavioral_file_tasks = tuple(t for t in canonical_tasks if t != "rest")

    initial_has: dict[str, set[str]] = {t: set() for t in canonical_tasks}
    for sub in all_subjects:
        for t in behavioral_file_tasks:
            if _subject_has_behavioral_raw(behavior_raw, sub, t):
                initial_has[t].add(sub)
        if _subject_has_rest_bids(bids_path, sub):
            initial_has["rest"].add(sub)

    initial_complete = set.intersection(*(initial_has[t] for t in canonical_tasks))

    final_has: dict[str, set[str]] = {}
    for t in canonical_tasks:
        final_has[t] = {
            s for s in initial_has[t] if (s, t) not in excluded_pairs
        }

    final_complete = set.intersection(*(final_has[t] for t in canonical_tasks))

    per_task: list[dict[str, object]] = []
    for t in canonical_tasks:
        n0 = len(initial_has[t])
        n1 = len(final_has[t])
        per_task.append(
            {
                "task": t,
                "n_with_task_before_exclusion": n0,
                "n_with_task_after_exclusion": n1,
                "n_dropped_for_task": n0 - n1,
            }
        )

    return {
        "exclusions_json_path": os.path.abspath(exclusions_json_path),
        "behavior_raw": os.path.abspath(behavior_raw),
        "bids_path": os.path.abspath(bids_path),
        "canonical_tasks": list(canonical_tasks),
        "total_subjects_n": len(all_subjects),
        "n_excluded_subject_task_pairs_in_json": len(excluded_pairs),
        "initial_complete_n_all_five_tasks": len(initial_complete),
        "final_complete_n_all_five_tasks": len(final_complete),
        "per_task": per_task,
    }


def format_exclusion_summary_text(summary: dict[str, object]) -> str:
    lines = [
        "UH2 AIM2 — exclusion sample-size summary",
        "",
        f"Exclusions file: {summary['exclusions_json_path']}",
        f"Behavioral raw root: {summary['behavior_raw']}",
        f"BIDS root: {summary['bids_path']}",
        "",
        "Definitions:",
        "  • Total subjects: unique subjects with a folder under behavioral raw, a BIDS sub-*,",
        "    or any subject listed in the exclusions JSON.",
        "  • “Has task” before exclusion: raw CSV exists for the four behavioral tasks;",
        "    rest = at least one *task-rest*_bold.nii* under sub-<id>/ (recursive).",
        "  • After exclusion: same, minus any (subject, task) triple-list entry in the JSON",
        "    that matches the five canonical tasks (case-insensitive).",
        "",
        f"Total number of subjects: {summary['total_subjects_n']}",
        f"Subject × task pairs flagged in JSON (canonical tasks only): "
        f"{summary['n_excluded_subject_task_pairs_in_json']}",
        "",
        "Complete five-task sample (has all five tasks before exclusions): "
        f"{summary['initial_complete_n_all_five_tasks']}",
        "Complete five-task sample (all five still kept after exclusions): "
        f"{summary['final_complete_n_all_five_tasks']}",
        "",
        "Per task (only among subjects who had that task before exclusion):",
        f"{'task':<22} {'N before':>10} {'N after':>10} {'Dropped':>10}",
    ]
    for row in summary["per_task"]:  # type: ignore[assignment]
        lines.append(
            f"{row['task']:<22} {row['n_with_task_before_exclusion']:>10} "
            f"{row['n_with_task_after_exclusion']:>10} {row['n_dropped_for_task']:>10}"
        )
    lines.append("")
    return "\n".join(lines)


def write_exclusion_summary_txt(
    exclusions_json_path: str,
    output_txt_path: str,
    *,
    behavior_raw: str | None = None,
    bids_path: str | None = None,
) -> dict[str, object]:
    """Write human-readable ``exclusions_summary.txt``; return the summary dict."""
    summary = compute_exclusion_sample_summary(
        exclusions_json_path,
        behavior_raw=behavior_raw,
        bids_path=bids_path,
    )
    txt_dir = os.path.dirname(os.path.abspath(output_txt_path))
    if txt_dir:
        os.makedirs(txt_dir, exist_ok=True)

    text = format_exclusion_summary_text(summary)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return summary
