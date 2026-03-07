#!/usr/bin/env python
"""Audit BIDS sidecar JSON metadata for flagged global-mean-signal subjects."""

from __future__ import annotations

import argparse
from pathlib import Path

from uh2_aim2.config import (
    BIDS_PATH,
    GLOBAL_MEAN_HIGH_SUBJECTS,
    SIDECAR_JSON_AUDIT_OUTPUT_DIR,
)
from uh2_aim2.utils.sidecar_json_audit_utils import (
    audit_flagged_subjects_vs_controls,
    load_sidecar_json_table,
)


def _parse_subjects(subjects_arg: str | None) -> list[int | str]:
    """Parse comma-separated subject ids."""
    if not subjects_arg:
        return list(GLOBAL_MEAN_HIGH_SUBJECTS)
    out: list[int | str] = []
    for token in subjects_arg.split(","):
        s = token.strip().replace("sub-", "").replace("s", "")
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError:
            out.append(s)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Systematically compare BIDS sidecar JSON metadata for flagged subjects "
            "against task-matched controls. JSON-only analysis (low compute)."
        )
    )
    parser.add_argument(
        "--bids-path",
        default=BIDS_PATH,
        help="Path to BIDS root.",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated flagged subject ids (default: GLOBAL_MEAN_HIGH_SUBJECTS from config).",
    )
    parser.add_argument(
        "--output-dir",
        default=SIDECAR_JSON_AUDIT_OUTPUT_DIR,
        help="Directory for audit output files.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.5,
        help="Z-score threshold for numeric key differences.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    flagged_subjects = _parse_subjects(args.subjects)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sidecar JSONs from: {args.bids_path}")
    sidecar_df = load_sidecar_json_table(args.bids_path)
    if sidecar_df.empty:
        print("No *_bold.json files found. Exiting.")
        return

    print(f"Loaded {len(sidecar_df)} sidecar rows")
    print(f"Flagged subjects: {flagged_subjects}")

    anomalies_df, summary_df = audit_flagged_subjects_vs_controls(
        sidecar_df=sidecar_df,
        flagged_subjects=flagged_subjects,
        z_threshold=args.z_threshold,
    )

    sidecar_table_path = output_dir / "sidecar_json_table_all_rows.csv"
    summary_path = output_dir / "sidecar_json_diff_summary.csv"
    anomalies_path = output_dir / "sidecar_json_flagged_anomalies.csv"

    sidecar_df.to_csv(sidecar_table_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    anomalies_df.to_csv(anomalies_path, index=False)

    print("\nSaved:")
    print(f"  - {sidecar_table_path}")
    print(f"  - {summary_path}")
    print(f"  - {anomalies_path}")

    if summary_df.empty:
        print("\nNo comparable task-level differences were detected.")
        return

    differing = summary_df[summary_df["different"] == True]
    print(f"\nDiffering task/key pairs: {len(differing)} / {len(summary_df)}")
    if not differing.empty:
        preview = differing[["task", "json_key", "data_type", "control_summary", "flagged_summary"]].head(20)
        print("\nTop differing keys:")
        print(preview.to_string(index=False))

    if not anomalies_df.empty:
        print(f"\nFlagged subject-run anomalies: {len(anomalies_df)}")
    else:
        print("\nNo row-level anomalies found in flagged subjects.")


if __name__ == "__main__":
    main()
