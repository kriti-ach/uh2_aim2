#!/bin/bash
#SBATCH --job-name=uh2_gms_qc
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Resolve repo directory from submit location (no hardcoded local paths)
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="$REPO_DIR/src/uh2_aim2/create_global_mean_signal_plots.py"
VENV_PY="$REPO_DIR/.venv/bin/python"

# -----------------------------------------------------------------------------
# Runtime controls (override with sbatch --export=...):
#   RUN_ALL_SUBJECTS=1 or 0
#   SUBJECT_ID=sub-XXXX (required only when RUN_ALL_SUBJECTS=0)
#   OUTPUT_MODE=png|both|pdf
# Notes:
#   - png  = create PNG(s) only
#   - both = create PNG(s) and PDF
#   - pdf  = same as both (kept for convenience)
#   - PDF generation is only supported for all-subject runs.
# Example usage:
#   sbatch --export=RUN_ALL_SUBJECTS=0,SUBJECT_ID=sub-1021,OUTPUT_MODE=png batch/run_global_mean_signal_plots.sbatch
#   sbatch --export=RUN_ALL_SUBJECTS=1,OUTPUT_MODE=pdf batch/run_global_mean_signal_plots.sbatch
# -----------------------------------------------------------------------------
RUN_ALL_SUBJECTS="${RUN_ALL_SUBJECTS:-1}"
SUBJECT_ID="${SUBJECT_ID:-}"
OUTPUT_MODE="${OUTPUT_MODE:-png}"

mkdir -p "$REPO_DIR/logs"

echo "[$(date)] Starting global mean signal plots job"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Repo dir: $REPO_DIR"

# Optional module loads for Stanford clusters (safe if unavailable)
module load gcc/10.1.0 2>/dev/null || true

# Non-interactive plotting backend and predictable thread usage
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ARGS=()
if [ "$RUN_ALL_SUBJECTS" -eq 1 ]; then
  ARGS+=(--all-subjects)
else
  if [ -z "$SUBJECT_ID" ]; then
    echo "ERROR: SUBJECT_ID must be set when RUN_ALL_SUBJECTS=0"
    exit 1
  fi
  ARGS+=(--subject "$SUBJECT_ID")
fi

case "$OUTPUT_MODE" in
  png)
    ;;
  both|pdf)
    if [ "$RUN_ALL_SUBJECTS" -ne 1 ]; then
      echo "ERROR: OUTPUT_MODE=$OUTPUT_MODE requires RUN_ALL_SUBJECTS=1"
      exit 1
    fi
    ARGS+=(--make-pdf)
    ;;
  *)
    echo "ERROR: OUTPUT_MODE must be one of: png, both, pdf"
    exit 1
    ;;
esac

echo "Run mode: RUN_ALL_SUBJECTS=$RUN_ALL_SUBJECTS SUBJECT_ID=${SUBJECT_ID:-NA} OUTPUT_MODE=$OUTPUT_MODE"
echo "Python args: ${ARGS[*]}"

if [ -x "$VENV_PY" ]; then
  "$VENV_PY" "$SCRIPT_PATH" "${ARGS[@]}"
else
  python "$SCRIPT_PATH" "${ARGS[@]}"
fi

echo "[$(date)] Finished global mean signal plots job"
