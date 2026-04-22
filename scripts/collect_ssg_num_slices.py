#!/usr/bin/env python
"""
Collect ``num_slices`` from JSON sidecars on Flywheel acquisitions whose session
label, acquisition label, or filename contains ``ssg`` (e.g. ``*_ssg_bold.json``).

``flywheel.Client()`` (env/session auth), ``fw.projects.find(group=..., label=...)``,
``fw.get_project_sessions``, ``fw.get_session_acquisitions``, and
``fw.download_file_from_acquisition`` when available.

Project defaults to ``russpold`` / ``uh2_aim2`` from ``uh2_aim2.config``. Skips ``qa.json`` / ``*_qa.json``.

Use ``--local`` to scan a directory tree instead (no API).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


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


def _qa_json(name: str) -> bool:
    n = name.lower()
    return n == "qa.json" or n.endswith("_qa.json")


def _iter_ssg_json_files(scan_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(scan_root.rglob("*.json")):
        rel = p.relative_to(scan_root).as_posix()
        if "ssg" not in rel.lower():
            continue
        if _qa_json(p.name):
            continue
        out.append(p)
    return out


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    """Match ``rdoc_fmri_quality_control`` session/acquisition attribute access."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _download_acquisition_file(fw: Any, acq_id: str, file_name: str, out_path: Path) -> None:
    """Same as ``rdoc_fmri_quality_control.scripts.run_flywheel_qc.download_acquisition_file``."""
    if hasattr(fw, "download_file_from_acquisition"):
        fw.download_file_from_acquisition(acq_id, file_name, str(out_path))
        return
    acq = fw.get_acquisition(acq_id)
    acq.download_file(file_name, str(out_path))


def _download_acquisition_json_via_fw(fw: Any, acq_id: str, filename: str) -> tuple[dict[str, Any] | None, str | None]:
    fd, path_str = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    path = Path(path_str)
    try:
        _download_acquisition_file(fw, acq_id, filename, path)
        with path.open(encoding="utf-8") as fp:
            return json.load(fp), None
    except Exception as e:
        return None, str(e)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def _ssg_context(session_label: str, acq_label: str, filename: str) -> bool:
    blob = f"{session_label} {acq_label} {filename}".lower()
    return "ssg" in blob


def collect_rows_from_flywheel(group_id: str, project_label: str) -> list[dict[str, object]]:
    import flywheel

    # Match rdoc_fmri_quality_control: rely on Flywheel auth already present in environment/session.
    fw = flywheel.Client()
    projects = fw.projects.find(f"group={group_id},label={project_label}")
    if not projects:
        raise RuntimeError(f"No project found for group={group_id}, label={project_label}")
    project = projects[0]

    rows: list[dict[str, object]] = []
    for ses in fw.get_project_sessions(project.id):
        ses_label = str(_obj_get(ses, "label", "") or "")
        for acq in fw.get_session_acquisitions(ses.id):
            acq_label = str(_obj_get(acq, "label", "") or "")
            files = _obj_get(acq, "files", None) or []
            acq_id = str(_obj_get(acq, "id", "") or "")
            if not acq_id:
                continue

            for finfo in files:
                name = str(_obj_get(finfo, "name", "") or "")
                if not name.lower().endswith(".json"):
                    continue
                if _qa_json(name):
                    continue
                if not _ssg_context(ses_label, acq_label, name):
                    continue

                rel = f"{ses_label}/{acq_label}/{name}".replace(" ", "_")
                data, err = _download_acquisition_json_via_fw(fw, acq_id, name)
                if err is not None or data is None:
                    rows.append(
                        {
                            "relative_path": rel,
                            "sub": "",
                            "ses": "",
                            "task": "",
                            "run": "",
                            "acq": "",
                            "json_file": name,
                            "num_slices": f"<read error: {err}>",
                        }
                    )
                    continue

                stem = Path(name).stem
                ent = _parse_bids_stem(stem)
                num = _num_slices_from_json(data)
                rows.append(
                    {
                        "relative_path": rel,
                        "sub": ent.get("sub", ""),
                        "ses": ent.get("ses", ""),
                        "task": ent.get("task", ""),
                        "run": ent.get("run", ""),
                        "acq": ent.get("acq", ""),
                        "json_file": name,
                        "num_slices": "" if num is None else num,
                    }
                )

    return rows


def collect_rows_from_local(scan_root: Path) -> list[dict[str, object]]:
    files = _iter_ssg_json_files(scan_root)
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
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export num_slices from Flywheel *ssg* JSON (API default) or from a local tree (--local)."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Scan a directory instead of Flywheel (uses --root or FLYWHEEL_JSON_EXPORT_PATH).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="With --local: directory to scan (default: UH2_FLYWHEEL_JSON_ROOT or ~/russpold/uh2_aim2).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        metavar="GROUP/LABEL",
        help="Override Flywheel project as group/project_label (default: russpold/uh2_aim2 from config).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: stdout).",
    )
    args = parser.parse_args()

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

    if args.local:
        from uh2_aim2.config import FLYWHEEL_JSON_EXPORT_PATH

        scan_root = args.root if args.root is not None else Path(FLYWHEEL_JSON_EXPORT_PATH)
        if not scan_root.is_dir():
            print(
                f"Not a directory: {scan_root}\n"
                "Set UH2_FLYWHEEL_JSON_ROOT or pass --root for --local mode.",
                file=sys.stderr,
            )
            return 1
        rows = collect_rows_from_local(scan_root)
    else:
        from uh2_aim2.config import FLYWHEEL_GROUP_ID, FLYWHEEL_PROJECT_LABEL

        if args.project:
            if "/" not in args.project:
                print("--project must be GROUP/LABEL (e.g. russpold/uh2_aim2)", file=sys.stderr)
                return 2
            group_id, project_label = args.project.split("/", 1)
        else:
            group_id, project_label = FLYWHEEL_GROUP_ID, FLYWHEEL_PROJECT_LABEL

        try:
            rows = collect_rows_from_flywheel(group_id, project_label)
        except Exception as e:
            print(
                f"Flywheel error: {e}\n"
                "Ensure flywheel-sdk is installed and Flywheel auth is configured in the environment "
                "(same as rdoc_fmri_quality_control), or use --local with a synced directory.",
                file=sys.stderr,
            )
            return 1

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
