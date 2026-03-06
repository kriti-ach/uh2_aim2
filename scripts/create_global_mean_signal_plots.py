#!/usr/bin/env python
"""Create global mean signal plots from BIDS NIfTI files."""

from __future__ import annotations

import argparse
from pathlib import Path

from uh2_aim2.config import (
    GLOBAL_MEAN_DEFAULT_SUBJECT,
    GLOBAL_MEAN_SIGNAL_BIDS_PATH,
    GLOBAL_MEAN_SIGNAL_OUTPUT_PDF,
    GLOBAL_MEAN_SIGNAL_OUTPUT_PNG_DIR,
)
from uh2_aim2.utils.global_mean_signal_utils import (
    create_global_mean_signal_pdf,
    create_global_mean_signal_png_for_subject,
    create_global_mean_signal_pngs_all_subjects,
)


def _first_subject_in_bids(bids_path: str) -> str:
    bids_root = Path(bids_path)
    subject_dirs = sorted([p for p in bids_root.glob("sub-*") if p.is_dir()], key=lambda p: p.name)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found under {bids_path}")
    return subject_dirs[0].name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create global mean signal plots. Default mode generates a single-subject PNG "
            "for fast QC. Use --all-subjects for one PNG per subject."
        )
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Generate one PNG per subject for all subjects.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=GLOBAL_MEAN_DEFAULT_SUBJECT,
        help="Subject ID for single-subject mode (e.g., sub-1021, s1021, or 1021).",
    )
    parser.add_argument(
        "--make-pdf",
        action="store_true",
        help="Also generate the multi-page PDF (slower).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.all_subjects:
        saved = create_global_mean_signal_pngs_all_subjects(
            bids_path=GLOBAL_MEAN_SIGNAL_BIDS_PATH,
            output_png_dir=GLOBAL_MEAN_SIGNAL_OUTPUT_PNG_DIR,
        )
        print(f"Saved {len(saved)} subject PNGs to: {GLOBAL_MEAN_SIGNAL_OUTPUT_PNG_DIR}")
    else:
        subject = args.subject or _first_subject_in_bids(GLOBAL_MEAN_SIGNAL_BIDS_PATH)
        output_path = create_global_mean_signal_png_for_subject(
            bids_path=GLOBAL_MEAN_SIGNAL_BIDS_PATH,
            output_png_dir=GLOBAL_MEAN_SIGNAL_OUTPUT_PNG_DIR,
            subject_id=subject,
        )
        print(f"Saved subject PNG: {output_path}")

    if args.make_pdf:
        create_global_mean_signal_pdf(
            bids_path=GLOBAL_MEAN_SIGNAL_BIDS_PATH,
            output_pdf_path=GLOBAL_MEAN_SIGNAL_OUTPUT_PDF,
        )
        print(f"Saved global mean signal report PDF: {GLOBAL_MEAN_SIGNAL_OUTPUT_PDF}")


if __name__ == "__main__":
    main()
