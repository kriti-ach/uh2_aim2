#!/usr/bin/env python
"""
Sanitize BIDS ``sub-*/func/*_events.tsv`` files (column drops + manipulation ``block_duration`` s→ms).

Default sample: subject 1021 only, output under ``$SCRATCH/uh2_aim2_events_sample`` (or ``-o``).

Examples::

    # Preview on Oak BIDS, write copy to scratch (default out dir + subject 1021)
    PYTHONPATH=src python scripts/sanitize_bids_events.py

    PYTHONPATH=src python scripts/sanitize_bids_events.py \\
        -o /scratch/users/kritiach/uh2_aim2_events_sample --subjects 1021

    # All subjects (writes to -o; Oak unchanged unless --apply-to-bids)
    PYTHONPATH=src python scripts/sanitize_bids_events.py -o /path/to/out --subjects all

    # Overwrite originals under BIDS (use with care). Either form works:
    PYTHONPATH=src python scripts/sanitize_bids_events.py --apply-to-bids --subjects all
    PYTHONPATH=src python scripts/sanitize_bids_events.py --apply-to-bids all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from uh2_aim2.config import BIDS_PATH
from uh2_aim2.utils.sanitize_events_utils import iter_subject_func_events, sanitize_events_file


def _default_output_root() -> str:
    base = os.environ.get("SCRATCH") or os.path.expanduser("~")
    return os.path.join(base, "uh2_aim2_events_sample")


def _list_all_subject_ids(bids_root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(bids_root.glob("sub-*")):
        if p.is_dir() and (p / "func").is_dir():
            out.append(p.name.replace("sub-", ""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bids-root",
        default=BIDS_PATH,
        help="BIDS root containing sub-*/func/*_events.tsv",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=_default_output_root(),
        help="Root for sanitized copies (mirrors sub-*/func/). Default uses $SCRATCH or ~.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="ID",
        help="Subject id(s), or ``all`` for every sub-* with func/. "
        "If omitted, use trailing IDs or default 1021.",
    )
    parser.add_argument(
        "--apply-to-bids",
        action="store_true",
        help="Overwrite event files in place under --bids-root.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Run summary CSV path (default: <output-dir>/sanitize_events_summary.csv)",
    )
    parser.add_argument(
        "subjects_trailing",
        nargs="*",
        metavar="ID",
        help="Optional subject id(s) or ``all`` (e.g. after --apply-to-bids). "
        "Ignored if --subjects is set.",
    )
    args = parser.parse_args()

    bids_root = Path(args.bids_root).resolve()
    if not bids_root.is_dir():
        print(f"BIDS root not found: {bids_root}", file=sys.stderr)
        return 1

    if args.subjects is not None:
        raw_subjects = list(args.subjects)
    elif args.subjects_trailing:
        raw_subjects = list(args.subjects_trailing)
    else:
        raw_subjects = ["1021"]

    if len(raw_subjects) == 1 and raw_subjects[0].lower() == "all":
        subject_ids = _list_all_subject_ids(bids_root)
    else:
        subject_ids = [
            str(s).strip().replace("sub-", "").replace("s", "") for s in raw_subjects
        ]

    if not subject_ids:
        print("No subjects to process.", file=sys.stderr)
        return 1

    if args.apply_to_bids:
        out_root = bids_root
        print("WARNING: --apply-to-bids overwrites original *_events.tsv files.", file=sys.stderr)
    else:
        out_root = Path(args.output_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else (out_root / "sanitize_events_summary.csv")
    )

    rows: list[dict[str, object]] = []
    for sid in subject_ids:
        paths = iter_subject_func_events(bids_root, sid)
        if not paths:
            print(f"  (no *_events.tsv under {bids_root / f'sub-{sid}' / 'func'})")
        for src in paths:
            dest, task, notes = sanitize_events_file(src, bids_root, out_root)
            print(f"  {src.name} -> {dest}")
            rows.append(
                {
                    "subject_id": sid,
                    "task": task or "",
                    "source": str(src),
                    "output": str(dest),
                    "notes": "; ".join(notes),
                }
            )

    summary_df = pd.DataFrame(rows)
    if not args.apply_to_bids:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary: {summary_path}")
    print(f"Files written: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
