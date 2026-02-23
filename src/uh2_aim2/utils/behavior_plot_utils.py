"""Plotting utilities for behavioral QC histograms."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import HISTOGRAM_BINS, HISTOGRAM_METRICS


def _resolve_metric(qc_df: pd.DataFrame, spec: str | tuple[str, str]) -> pd.Series:
    """Extract a metric column or compute a difference between two columns.

    Args:
        qc_df: Per-task QC DataFrame (subjects as index, no mean/std rows).
        spec: Either a column name or a (col_a, col_b) tuple for col_a − col_b.
    """
    if isinstance(spec, tuple):
        col_a, col_b = spec
        return qc_df[col_a] - qc_df[col_b]
    return qc_df[spec]


def plot_task_histograms(
    qc_df: pd.DataFrame,
    task: str,
    output_path: str,
) -> str | None:
    """Generate a single PNG with one histogram per metric for a task.

    Args:
        qc_df: Per-task QC DataFrame (with mean/std rows included).
        task: Task name (must be a key in HISTOGRAM_METRICS).
        output_path: Directory to save the PNG.

    Returns:
        Path to the saved PNG, or None if no metrics defined.
    """
    metrics = HISTOGRAM_METRICS.get(task)
    if not metrics:
        return None

    # Drop summary rows
    df = qc_df.drop(index=["mean", "std"], errors="ignore")

    n_metrics = len(metrics)
    n_cols = min(n_metrics, 3)
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for idx, (spec, label) in enumerate(metrics):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        try:
            values = _resolve_metric(df, spec).dropna()
        except KeyError:
            ax.set_title(f"{label}\n(column not found)", fontsize=10)
            ax.set_visible(True)
            continue

        if values.empty:
            ax.set_title(f"{label}\n(no data)", fontsize=10)
            continue

        ax.hist(values, bins=HISTOGRAM_BINS, edgecolor="white", alpha=0.85)
        ax.axvline(values.mean(), color="red", linestyle="--", linewidth=1, label=f"mean = {values.mean():.3f}")
        ax.axvline(values.median(), color="orange", linestyle=":", linewidth=1, label=f"median = {values.median():.3f}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"{task} — QC Metric Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    filepath = os.path.join(output_path, f"{task}_histograms.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


def plot_all_task_histograms(
    qc_results: dict[str, pd.DataFrame],
    output_path: str,
) -> list[str]:
    """Generate histogram PNGs for all tasks that have specs defined.

    Args:
        qc_results: Dict mapping task name → per-task QC DataFrame.
        output_path: Directory to save the PNGs.

    Returns:
        List of paths to saved PNGs.
    """
    saved = []

    for task, qc_df in qc_results.items():
        filepath = plot_task_histograms(qc_df, task, output_path)
        if filepath:
            saved.append(filepath)
            print(f"  {os.path.basename(filepath)}")

    return saved
