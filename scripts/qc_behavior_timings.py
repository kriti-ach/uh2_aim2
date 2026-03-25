#!/usr/bin/env python
"""
Check scanner / fMRI trigger wait spans in raw behavioral CSVs under BEHAVIOR_DATA.

Writes a CSV listing subject, task, observed vs expected duration, and flags.
"""

import argparse
import os
import sys

from uh2_aim2.config import BEHAVIOR_TIMING_QC_CSV, BEHAVIOR_TIMING_TOLERANCE_S, TASKS
from uh2_aim2.utils.behavior_timing_qc_utils import run_behavior_timing_qc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject folder names (e.g. 1021). Default: all under BEHAVIOR_DATA with task/",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Tasks to check. Default: {TASKS}",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=BEHAVIOR_TIMING_TOLERANCE_S,
        help=f"Absolute tolerance in seconds (default {BEHAVIOR_TIMING_TOLERANCE_S})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=BEHAVIOR_TIMING_QC_CSV,
        help="Output CSV path",
    )
    args = parser.parse_args()

    df = run_behavior_timing_qc(
        subjects=list(args.subjects) if args.subjects else None,
        tasks=list(args.tasks) if args.tasks else None,
        tolerance_s=args.tolerance,
    )
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_flag = int((~df["ok"]).sum())
    print(f"Wrote {args.output} ({len(df)} rows, {n_flag} flagged)")
    return 1 if n_flag else 0


if __name__ == "__main__":
    sys.exit(main())
