#!/usr/bin/env python
"""Audit Flywheel pfile metadata (aps_r1/aps_r2/aps_tg) for flagged subjects."""

from __future__ import annotations

import argparse

from uh2_aim2.config import (
    FLYWHEEL_PROJECT_PATH,
    GLOBAL_MEAN_HIGH_SUBJECTS,
    PFILE_METADATA_AUDIT_OUTPUT_DIR,
)
from uh2_aim2.utils.flywheel_pfile_audit_utils import (
    collect_flagged_subject_pfile_metadata,
    save_pfile_audit_outputs,
    summarize_gain_differences,
)


def _parse_subject_list(subjects_arg: str | None) -> list[int | str]:
    if not subjects_arg:
        return list(GLOBAL_MEAN_HIGH_SUBJECTS)
    out: list[int | str] = []
    for token in subjects_arg.split(","):
        s = token.strip()
        if not s:
            continue
        s = s.replace("sub-", "").replace("s", "")
        try:
            out.append(int(s))
        except ValueError:
            out.append(s)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query Flywheel metadata only (no pfile downloads) for aps_r1/aps_r2/aps_tg "
            "in pfile metadata for flagged subjects."
        )
    )
    parser.add_argument(
        "--project-path",
        default=FLYWHEEL_PROJECT_PATH,
        help="Flywheel project path (e.g., russpold/uh2aim2).",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject ids. Defaults to GLOBAL_MEAN_HIGH_SUBJECTS from config.",
    )
    parser.add_argument(
        "--output-dir",
        default=PFILE_METADATA_AUDIT_OUTPUT_DIR,
        help="Output directory for CSV reports.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    flagged_subjects = _parse_subject_list(args.subjects)

    try:
        import flywheel
    except ImportError as exc:
        raise ImportError(
            "flywheel SDK is required. Install it in your environment: pip install flywheel-sdk"
        ) from exc

    fw = flywheel.Client()  # uses env/auth config on cluster

    print(f"Project: {args.project_path}")
    print(f"Flagged subjects: {flagged_subjects}")
    print("Collecting pfile metadata from Flywheel (no file downloads)...")

    detailed_df = collect_flagged_subject_pfile_metadata(
        fw_client=fw,
        project_path=args.project_path,
        flagged_subjects=flagged_subjects,
    )
    summary_df = summarize_gain_differences(detailed_df)
    detailed_path, summary_path = save_pfile_audit_outputs(
        detailed_df=detailed_df,
        summary_df=summary_df,
        output_dir=args.output_dir,
    )

    print(f"\nSaved detailed report: {detailed_path}")
    print(f"Saved summary report:  {summary_path}")
    if detailed_df.empty:
        print("\nNo pfile metadata rows found for requested subjects.")
        return

    found_tasks = sorted([t for t in detailed_df["task"].dropna().unique().tolist()])
    print(f"\nDetected tasks in scan names: {found_tasks}")
    unstable = summary_df[summary_df["possible_gain_instability"] == True]
    print(f"Subject/task entries with potential gain instability: {len(unstable)}")
    if not unstable.empty:
        print(
            unstable[
                [
                    "subject_id",
                    "task",
                    "aps_r1_values",
                    "aps_r2_values",
                    "aps_tg_values",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
