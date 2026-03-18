#!/bin/bash
#SBATCH --job-name=uh2_pfile_meta
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="$REPO_DIR/scripts/audit_flywheel_pfile_metadata.py"
VENV_PY="$REPO_DIR/.venv/bin/python"

mkdir -p "$REPO_DIR/logs"

echo "[$(date)] Starting Flywheel pfile metadata audit"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Repo dir: $REPO_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -x "$VENV_PY" ]; then
  "$VENV_PY" "$SCRIPT_PATH"
else
  python "$SCRIPT_PATH"
fi

echo "[$(date)] Finished Flywheel pfile metadata audit"
