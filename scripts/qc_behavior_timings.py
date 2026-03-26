#!/usr/bin/env python
"""
Check scanner / fMRI trigger wait spans in raw behavioral CSVs under BEHAVIOR_DATA.

Writes a CSV listing subject, task, observed vs expected duration, and flags.
"""

import argparse
import os
import sys

from uh2_aim2.config import (
    BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD,
    BEHAVIOR_TIMING_QC_CSV,
    BEHAVIOR_TIMING_QC_FLAGGED_CSV,
    TASKS,
)
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
        "-o",
        "--output",
        default=BEHAVIOR_TIMING_QC_CSV,
        help="Output CSV path (all rows)",
    )
    parser.add_argument(
        "--flagged-output",
        default=BEHAVIOR_TIMING_QC_FLAGGED_CSV,
        help="Output CSV path (only flagged rows with delta above threshold)",
    )
    args = parser.parse_args()

    df = run_behavior_timing_qc(
        subjects=list(args.subjects) if args.subjects else None,
        tasks=list(args.tasks) if args.tasks else None,
    )
    out_dir = os.path.dirname(os.path.abspath(args.output))
    flagged_out_dir = os.path.dirname(os.path.abspath(args.flagged_output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if flagged_out_dir:
        os.makedirs(flagged_out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    flagged_df = df[(~df["ok"]) & (df["delta"] >= BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD)].copy()
    flagged_df.to_csv(args.flagged_output, index=False)

    n_flag = int((~df["ok"]).sum())
    print(f"Wrote {args.output} ({len(df)} rows, {n_flag} flagged)")
    print(
        f"Wrote {args.flagged_output} "
        f"({len(flagged_df)} rows; delta >= {BEHAVIOR_TIMING_FLAG_DELTA_THRESHOLD})"
    )
    return 1 if n_flag else 0


if __name__ == "__main__":
    sys.exit(main())
