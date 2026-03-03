#!/bin/bash
#SBATCH --job-name=uh2_gms_qc
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Resolve repo directory from submit location (no hardcoded local paths)
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_PATH="$REPO_DIR/src/uh2_aim2/create_global_mean_signal_plots.py"
VENV_PY="$REPO_DIR/.venv/bin/python"

mkdir -p "$REPO_DIR/logs"

echo "[$(date)] Starting global mean signal PDF job"
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

if [ -x "$VENV_PY" ]; then
  "$VENV_PY" "$SCRIPT_PATH"
else
  python "$SCRIPT_PATH"
fi

echo "[$(date)] Finished global mean signal PDF job"
