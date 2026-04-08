#!/usr/bin/env python
"""
Audit BIDS events.tsv files for schema inconsistencies and likely unnecessary columns.

Focus tasks:
- discountFix
- stopSignal
- motorSelectiveStop
- manipulationTask
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


TARGET_TASKS = {"discountFix", "stopSignal", "motorSelectiveStop", "manipulationTask"}
TASK_RE = re.compile(r"task-([A-Za-z0-9]+)_")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        cols = rdr.fieldnames or []
        rows = list(rdr)
    return cols, rows


def _as_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        default="/oak/stanford/groups/russpold/data/uh2/aim2/BIDS",
        help="BIDS root containing sub-*/func/*events.tsv",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_outputs/event_file_audit",
        help="Directory for output CSV summaries",
    )
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(bids_root.glob("sub-*/func/*events.tsv"))
    if not files:
        print(f"No events.tsv files found under: {bids_root}")
        return 1

    per_file_flags: list[dict[str, object]] = []
    per_task_col_presence: dict[str, Counter[str]] = defaultdict(Counter)
    per_task_file_count: Counter[str] = Counter()
    per_task_schema: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    per_task_block_unit_votes: dict[str, Counter[str]] = defaultdict(Counter)

    for p in files:
        m = TASK_RE.search(p.name)
        if not m:
            continue
        task = m.group(1)
        if task not in TARGET_TASKS:
            continue

        subject = p.parts[-3]  # sub-xxxx
        cols, rows = _read_tsv(p)
        per_task_file_count[task] += 1
        per_task_schema[task][tuple(cols)] += 1
        for c in cols:
            per_task_col_presence[task][c] += 1

        # discountFix trial_id availability
        if task == "discountFix":
            if "trial_id" not in cols:
                per_file_flags.append(
                    {
                        "subject": subject,
                        "task": task,
                        "file": str(p),
                        "flag_type": "missing_column",
                        "column": "trial_id",
                        "details": "trial_id column missing",
                    }
                )
            else:
                nonempty = False
                for r in rows:
                    v = (r.get("trial_id") or "").strip()
                    if v and v.upper() != "N/A" and v.lower() != "nan":
                        nonempty = True
                        break
                if not nonempty:
                    per_file_flags.append(
                        {
                            "subject": subject,
                            "task": task,
                            "file": str(p),
                            "flag_type": "column_uninformative",
                            "column": "trial_id",
                            "details": "trial_id present but always empty/N/A",
                        }
                    )

        # worker_id formatting consistency
        if "worker_id" in cols:
            pref_s = 0
            no_s = 0
            for r in rows:
                v = (r.get("worker_id") or "").strip()
                if not v:
                    continue
                if v.lower().startswith("s"):
                    pref_s += 1
                else:
                    no_s += 1
            if pref_s and no_s:
                per_file_flags.append(
                    {
                        "subject": subject,
                        "task": task,
                        "file": str(p),
                        "flag_type": "format_inconsistency",
                        "column": "worker_id",
                        "details": "mix of worker_id values with and without leading 's'",
                    }
                )

        # junk column checks
        if "junk" in cols:
            junk_vals = [(r.get("junk") or "").strip().lower() for r in rows]
            if rows:
                last = junk_vals[-1]
                if last in {"1", "true", "yes", "junk"}:
                    per_file_flags.append(
                        {
                            "subject": subject,
                            "task": task,
                            "file": str(p),
                            "flag_type": "review_needed",
                            "column": "junk",
                            "details": "last row marked as junk",
                        }
                    )
            informative = {v for v in junk_vals if v not in {"", "0", "false", "no", "nan", "n/a"}}
            if not informative:
                per_file_flags.append(
                    {
                        "subject": subject,
                        "task": task,
                        "file": str(p),
                        "flag_type": "column_uninformative",
                        "column": "junk",
                        "details": "junk column appears unused (always false/empty)",
                    }
                )

        # block_duration units check
        if "block_duration" in cols:
            vals = []
            for r in rows:
                x = _as_float((r.get("block_duration") or "").strip())
                if x is not None:
                    vals.append(x)
            if vals:
                ms_like = sum(1 for x in vals if x > 100)
                s_like = sum(1 for x in vals if 0 < x <= 100)
                unit = "mixed"
                if ms_like and not s_like:
                    unit = "ms"
                elif s_like and not ms_like:
                    unit = "s"
                per_task_block_unit_votes[task][unit] += 1
                if unit == "mixed":
                    per_file_flags.append(
                        {
                            "subject": subject,
                            "task": task,
                            "file": str(p),
                            "flag_type": "unit_inconsistency_within_file",
                            "column": "block_duration",
                            "details": "block_duration has both ms-like and s-like values",
                        }
                    )

    # Cross-file / cross-task summaries
    summary_rows: list[dict[str, object]] = []
    for task in sorted(per_task_file_count):
        n_files = per_task_file_count[task]
        schema_variants = len(per_task_schema[task])
        block_unit_votes = dict(per_task_block_unit_votes[task])
        summary_rows.append(
            {
                "task": task,
                "n_files": n_files,
                "schema_variants": schema_variants,
                "block_duration_unit_votes": str(block_unit_votes),
            }
        )

        # columns absent in at least one file
        for col, n in per_task_col_presence[task].items():
            if n < n_files:
                per_file_flags.append(
                    {
                        "subject": "*",
                        "task": task,
                        "file": "*",
                        "flag_type": "schema_inconsistency_across_files",
                        "column": col,
                        "details": f"column appears in {n}/{n_files} files",
                    }
                )

        # columns present in all files but likely not useful (all empty or constant)
        # sampled by scanning again for stability
        cols_all = [c for c, n in per_task_col_presence[task].items() if n == n_files]
        task_files = [p for p in files if TASK_RE.search(p.name) and TASK_RE.search(p.name).group(1) == task]
        for col in cols_all:
            values = []
            for p in task_files:
                _, rows = _read_tsv(p)
                for r in rows:
                    values.append((r.get(col) or "").strip())
            nonempty = [v for v in values if v and v.lower() not in {"nan", "n/a"}]
            unique_nonempty = set(nonempty)
            if len(nonempty) == 0:
                per_file_flags.append(
                    {
                        "subject": "*",
                        "task": task,
                        "file": "*",
                        "flag_type": "possibly_unnecessary_column",
                        "column": col,
                        "details": "column empty/N/A across all files",
                    }
                )
            elif len(unique_nonempty) == 1:
                per_file_flags.append(
                    {
                        "subject": "*",
                        "task": task,
                        "file": "*",
                        "flag_type": "possibly_unnecessary_column",
                        "column": col,
                        "details": f"column constant across all files: {next(iter(unique_nonempty))}",
                    }
                )

    # Cross-task block_duration unit inconsistency
    unit_by_task = {}
    for task, votes in per_task_block_unit_votes.items():
        if not votes:
            continue
        unit_by_task[task] = votes.most_common(1)[0][0]
    if len(set(unit_by_task.values())) > 1:
        per_file_flags.append(
            {
                "subject": "*",
                "task": "*",
                "file": "*",
                "flag_type": "unit_inconsistency_across_tasks",
                "column": "block_duration",
                "details": f"dominant units by task: {unit_by_task}",
            }
        )

    flags_csv = outdir / "event_file_inconsistency_flags.csv"
    summary_csv = outdir / "event_file_task_summary.csv"

    with flags_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["subject", "task", "file", "flag_type", "column", "details"],
        )
        w.writeheader()
        w.writerows(per_file_flags)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["task", "n_files", "schema_variants", "block_duration_unit_votes"],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote: {flags_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Flags: {len(per_file_flags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

