#!/usr/bin/env python
"""
List ``num_slices`` (and file paths) for JSON sidecars from a **Flywheel export**
tree (not Oak BIDS). Any relative path containing ``ssg`` is included, e.g.
``...task-motorSelectiveStop_run-1_ssg...``. Skips ``qa.json`` / ``*_qa.json``.

Default root: env ``UH2_FLYWHEEL_JSON_ROOT``, else ``~/flywheel/russpold/uh2_aim2`` (see ``config.FLYWHEEL_*``).

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


def _iter_ssg_json_files(scan_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(scan_root.rglob("*.json")):
        rel = p.relative_to(scan_root).as_posix()
        if "ssg" not in rel.lower():
            continue
        name = p.name.lower()
        if name == "qa.json" or name.endswith("_qa.json"):
            continue
        out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export num_slices from Flywheel *ssg* JSON (excluding qa.json) to CSV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Flywheel export directory (default: UH2_FLYWHEEL_JSON_ROOT or ~/flywheel/russpold/uh2_aim2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: stdout).",
    )
    args = parser.parse_args()

    if args.root is not None:
        scan_root = args.root
    else:
        from uh2_aim2.config import FLYWHEEL_JSON_EXPORT_PATH

        scan_root = Path(FLYWHEEL_JSON_EXPORT_PATH)

    if not scan_root.is_dir():
        print(
            f"Not a directory: {scan_root}\n"
            "Set UH2_FLYWHEEL_JSON_ROOT or pass --root to your Flywheel export for russpold/uh2_aim2.",
            file=sys.stderr,
        )
        return 1

    files = _iter_ssg_json_files(scan_root)
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
        rel = p.relative_to(scan_root).as_posix()
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
