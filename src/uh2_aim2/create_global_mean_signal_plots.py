#!/usr/bin/env python
"""Create one multi-page global mean signal PDF from BIDS NIfTI files."""

from config import GLOBAL_MEAN_SIGNAL_BIDS_PATH, GLOBAL_MEAN_SIGNAL_OUTPUT_PDF
from utils.global_mean_signal_utils import create_global_mean_signal_pdf


def main() -> None:
    create_global_mean_signal_pdf(
        bids_path=GLOBAL_MEAN_SIGNAL_BIDS_PATH,
        output_pdf_path=GLOBAL_MEAN_SIGNAL_OUTPUT_PDF,
    )
    print(f"Saved global mean signal report: {GLOBAL_MEAN_SIGNAL_OUTPUT_PDF}")


if __name__ == "__main__":
    main()
