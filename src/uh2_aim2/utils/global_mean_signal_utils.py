"""Utilities to create global mean signal QC plots from BIDS NIfTI files."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from config import (
    GLOBAL_MEAN_PANEL2_THRESHOLD,
    GLOBAL_MEAN_SHADE_END_TR,
    GLOBAL_MEAN_SHADE_START_TR,
    GLOBAL_MEAN_TASK_ORDER,
)

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nibabel is required for global mean signal plotting. "
        "Install with `pip install nibabel`."
    ) from exc


TASK_PATTERN = re.compile(r"task-([A-Za-z0-9]+)")


def _extract_task_name(nifti_path: Path) -> str | None:
    """Extract BIDS task label from filename."""
    match = TASK_PATTERN.search(nifti_path.name)
    if not match:
        return None
    return match.group(1)


def _compute_run_metrics(nifti_path: Path) -> dict[str, np.ndarray]:
    """Compute per-TR metrics for one 4D run."""
    img = nib.load(str(nifti_path))  # type: ignore[attr-defined]
    data = img.get_fdata(dtype=np.float32)  # type: ignore[attr-defined]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape {data.shape} for {nifti_path}")

    # data shape: X, Y, Z, T
    t = data.shape[-1]

    # Global mean signal across all voxels
    flat = data.reshape(-1, t)
    global_mean = flat.mean(axis=0)

    # Delta from first TR
    delta_from_tr1 = global_mean - global_mean[0]

    # Slice-wise std dev:
    # For each TR, compute mean signal per z-slice then std across slices.
    slice_means = data.mean(axis=(0, 1))  # shape (Z, T)
    slice_std = slice_means.std(axis=0)

    # DVARS: RMS of voxel-wise derivative
    diff = np.diff(flat, axis=1)
    dvars = np.sqrt(np.mean(diff * diff, axis=0))
    dvars = np.insert(dvars, 0, 0.0)  # align to TR index

    return {
        "global_mean": global_mean,
        "delta_from_tr1": delta_from_tr1,
        "slice_std": slice_std,
        "dvars": dvars,
    }


def _average_metric_across_runs(run_arrays: list[np.ndarray]) -> np.ndarray:
    """Average metric across runs after truncating to common minimum TR length."""
    if not run_arrays:
        return np.array([], dtype=np.float32)

    min_len = min(arr.shape[0] for arr in run_arrays)
    stacked = np.vstack([arr[:min_len] for arr in run_arrays])
    return stacked.mean(axis=0)


def _subject_task_metrics(subject_func_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Collect and average run metrics by task for one subject across sessions."""
    task_runs: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )

    nifti_files = sorted(subject_func_dir.rglob("*_bold.nii.gz"))
    for nifti_path in nifti_files:
        task = _extract_task_name(nifti_path)
        if not task:
            continue
        run_metrics = _compute_run_metrics(nifti_path)
        for metric_name, arr in run_metrics.items():
            task_runs[task][metric_name].append(arr)

    subject_metrics: dict[str, dict[str, np.ndarray]] = {}
    for task, metric_dict in task_runs.items():
        subject_metrics[task] = {
            metric_name: _average_metric_across_runs(arrays)
            for metric_name, arrays in metric_dict.items()
        }

    return subject_metrics


def _normalize_subject_label(subject_id: str) -> str:
    """Normalize user-provided subject id to BIDS style 'sub-<id>'."""
    subj = subject_id.strip()
    if subj.startswith("sub-"):
        return subj
    if subj.startswith("s"):
        return f"sub-{subj[1:]}"
    return f"sub-{subj}"


def _list_subject_dirs(bids_root: Path) -> list[Path]:
    """List subject directories under BIDS root."""
    return sorted(
        [path for path in bids_root.glob("sub-*") if path.is_dir()],
        key=lambda path: path.name,
    )


def _ordered_tasks(task_metrics: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Return tasks in configured order, then any extras."""
    present = set(task_metrics.keys())
    ordered = [task for task in GLOBAL_MEAN_TASK_ORDER if task in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _plot_subject_page(
    subject_id: str,
    task_metrics: dict[str, dict[str, np.ndarray]],
) -> Figure:
    """Create a 4-panel QC figure for one subject."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("#e5e5e5")

    tasks = _ordered_tasks(task_metrics)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(tasks))))

    # Panel 1: global mean
    # Panel 2: delta from TR1
    # Panel 3: slice-wise std dev
    # Panel 4: DVARS
    panel_map = [
        ("global_mean", "Global Mean Signal"),
        ("delta_from_tr1", "Mean Signal (TR[n] - TR[1])"),
        ("slice_std", "Slice-wise Std Dev"),
        ("dvars", "DVARS"),
    ]

    for color, task in zip(colors, tasks):
        metrics = task_metrics[task]
        for ax, (metric_name, y_label) in zip(axes, panel_map):
            y = metrics.get(metric_name, np.array([], dtype=np.float32))
            if y.size == 0:
                continue
            x = np.arange(y.shape[0])
            ax.plot(x, y, marker="o", markersize=2, linewidth=1, label=task, color=color)
            ax.set_ylabel(y_label)

    # Add panel-2 thresholds
    axes[1].axhline(
        GLOBAL_MEAN_PANEL2_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"+{GLOBAL_MEAN_PANEL2_THRESHOLD:g} threshold",
    )
    axes[1].axhline(
        -GLOBAL_MEAN_PANEL2_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"-{GLOBAL_MEAN_PANEL2_THRESHOLD:g} threshold",
    )

    # Add shaded dummy TR region to all panels
    for ax in axes:
        ax.axvspan(
            GLOBAL_MEAN_SHADE_START_TR,
            GLOBAL_MEAN_SHADE_END_TR,
            color="#eec5cf",
            alpha=0.35,
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("TR index")
    fig.suptitle(subject_id, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def create_global_mean_signal_png_for_subject(
    bids_path: str,
    output_png_dir: str,
    subject_id: str,
) -> Path:
    """Create one PNG for a single subject."""
    bids_root = Path(bids_path)
    output_dir = Path(output_png_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_label = _normalize_subject_label(subject_id)
    subject_dir = bids_root / subject_label
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    task_metrics = _subject_task_metrics(subject_dir)
    if not task_metrics:
        raise ValueError(f"No task NIfTI data found for {subject_label}")

    fig = _plot_subject_page(subject_label, task_metrics)
    output_path = output_dir / f"{subject_label}_global_mean_signal.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_global_mean_signal_pngs_all_subjects(
    bids_path: str,
    output_png_dir: str,
) -> list[Path]:
    """Create one PNG per subject for all subjects in BIDS."""
    bids_root = Path(bids_path)
    output_dir = Path(output_png_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for subject_dir in _list_subject_dirs(bids_root):
        task_metrics = _subject_task_metrics(subject_dir)
        if not task_metrics:
            continue
        fig = _plot_subject_page(subject_dir.name, task_metrics)
        output_path = output_dir / f"{subject_dir.name}_global_mean_signal.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def create_global_mean_signal_pdf(bids_path: str, output_pdf_path: str) -> None:
    """Create one large PDF with one QC page per subject from BIDS NIfTIs."""
    bids_root = Path(bids_path)
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_dirs = _list_subject_dirs(bids_root)

    with PdfPages(output_path) as pdf:
        for subject_dir in subject_dirs:
            # Use whole subject dir so this works with or without ses-* layer
            task_metrics = _subject_task_metrics(subject_dir)
            if not task_metrics:
                continue

            fig = _plot_subject_page(subject_dir.name, task_metrics)
            pdf.savefig(fig)
            plt.close(fig)
