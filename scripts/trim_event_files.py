#!/usr/bin/env python
"""Trim configured BIDS ``*events.tsv`` files (see ``BIDS_EVENT_FILES_TO_TRIM``)."""

from __future__ import annotations

import argparse

from uh2_aim2.config import TRIMMED_EVENT_OUTPUT_ROOT
from uh2_aim2.utils.event_files_utils import trim_configured_event_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trim event files at the first row where key_press == -1 and all remaining "
            "key_press values are -1. Always writes preview outputs to "
            "trimmed_event_file_outputs; optionally applies changes to source files."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Also overwrite original files under BIDS_PATH.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary_df = trim_configured_event_files(apply_to_source=args.apply)

    print("=" * 72)
    print("Trim Event Files")
    print("=" * 72)
    print(f"Preview output root: {TRIMMED_EVENT_OUTPUT_ROOT}")
    print(f"Apply to source: {args.apply}")
    print(f"Files processed: {len(summary_df)}")
    if not summary_df.empty:
        print(f"Files trimmed: {int(summary_df['trim_applied'].sum())}")
        print(f"Rows removed total: {int(summary_df['rows_removed'].sum())}")
    print("=" * 72)


if __name__ == "__main__":
    main()
