#!/usr/bin/env python
"""
List ``num_slices`` (and file paths) for JSON sidecars under BIDS paths that
contain ``ssg`` in the relative path (e.g. ``...task-motorSelectiveStop_run-1_ssg...``).
Skips ``qa.json`` / ``*_qa.json``.

Writes CSV: relative_path, subject, task, run, acq, num_slices
(some entity columns may be empty if the filename is nonstandard).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def _parse_bids_stem(stem: str) -> dict[str, str]:
    """Parse ``sub-*, ses-*, task-*, acq-*, run-*`` from a BIDS-like stem."""
    out: dict[str, str] = {}
    for key in ("sub", "ses", "task", "acq", "run"):
        m = re.search(rf"{key}-([^\s_]+)", stem)
        if m:
            out[key] = m.group(1)
    return out


def _num_slices_from_json(data: object) -> object:
    if not isinstance(data, dict):
        return None
    for key in ("num_slices", "NumberOfSlices", "SliceNumber"):
        v = data.get(key)
        if v is not None:
            return v
    return None


def _iter_ssg_json_files(bids_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(bids_root.rglob("*.json")):
        rel = p.relative_to(bids_root).as_posix()
        if "ssg" not in rel.lower():
            continue
        name = p.name.lower()
        if name == "qa.json" or name.endswith("_qa.json"):
            continue
        out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export num_slices from *ssg* JSON (excluding qa.json) to CSV."
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=None,
        help="BIDS root (default: uh2_aim2.config.BIDS_PATH)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: stdout).",
    )
    args = parser.parse_args()

    if args.bids_root is None:
        from uh2_aim2.config import BIDS_PATH

        bids_root = Path(BIDS_PATH)
    else:
        bids_root = args.bids_root

    if not bids_root.is_dir():
        print(f"Not a directory: {bids_root}", file=sys.stderr)
        return 1

    files = _iter_ssg_json_files(bids_root)
    fieldnames = [
        "relative_path",
        "sub",
        "ses",
        "task",
        "run",
        "acq",
        "json_file",
        "num_slices",
    ]
    rows: list[dict[str, object]] = []

    for p in files:
        rel = p.relative_to(bids_root).as_posix()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            rows.append(
                {
                    "relative_path": rel,
                    "sub": "",
                    "ses": "",
                    "task": "",
                    "run": "",
                    "acq": "",
                    "json_file": p.name,
                    "num_slices": f"<read error: {e}>",
                }
            )
            continue

        ent = _parse_bids_stem(p.stem)
        num = _num_slices_from_json(data)
        rows.append(
            {
                "relative_path": rel,
                "sub": ent.get("sub", ""),
                "ses": ent.get("ses", ""),
                "task": ent.get("task", ""),
                "run": ent.get("run", ""),
                "acq": ent.get("acq", ""),
                "json_file": p.name,
                "num_slices": "" if num is None else num,
            }
        )

    if args.output is None:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
