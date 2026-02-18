"""UH2 AIM2 Behavioral Analysis Package."""

from uh2_aim2.behavior_qc.behavior_qc import run_qc_pipeline

__all__ = ["run_qc_pipeline"]


def main() -> None:
    """Run the behavioral QC pipeline."""
    run_qc_pipeline()
