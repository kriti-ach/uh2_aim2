#!/usr/bin/env python
"""
Summarize sample size before vs after ``exclusions.json`` (behavioral, fMRIPrep, other).

Writes ``exclusions_summary.txt`` in the output directory (default: directory of the JSON).

Total subjects = union of subjects under behavioral raw, BIDS ``sub-*``, or mentioned in the JSON.
Initial “has task” = raw behavioral CSV (four tasks) or BIDS ``*task-rest*_bold.nii*``.

Usage::

    PYTHONPATH=src python scripts/summarize_exclusions.py

    PYTHONPATH=src python scripts/summarize_exclusions.py \\
        --exclusions-json /path/to/exclusions.json -o /path/to/outdir
"""

from __future__ import annotations

import argparse
import os
import sys

from uh2_aim2.config import BEHAVIOR_DATA_RAW, BIDS_PATH, FINAL_EXCLUSIONS_JSON_PATH
from uh2_aim2.utils.exclusion_summary_utils import (
    format_exclusion_summary_text,
    write_exclusion_summary_txt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclusions-json",
        default=FINAL_EXCLUSIONS_JSON_PATH,
        help="Path to exclusions.json",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for exclusions_summary.txt (default: same directory as exclusions JSON)",
    )
    parser.add_argument(
        "--behavior-raw",
        default=BEHAVIOR_DATA_RAW,
        help="Behavioral raw root",
    )
    parser.add_argument(
        "--bids",
        default=BIDS_PATH,
        help="BIDS root",
    )
    args = parser.parse_args()

    excl = os.path.abspath(args.exclusions_json)
    if not os.path.isfile(excl):
        print(f"Exclusions file not found: {excl}", file=sys.stderr)
        return 1

    out_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else (os.path.dirname(excl) or ".")
    )
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, "exclusions_summary.txt")

    summary = write_exclusion_summary_txt(
        excl,
        txt_path,
        behavior_raw=args.behavior_raw,
        bids_path=args.bids,
    )
    print(format_exclusion_summary_text(summary))
    print(f"Wrote: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
