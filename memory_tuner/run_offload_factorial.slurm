#!/usr/bin/env bash
#SBATCH --job-name=rlvram-offload-factorial
#SBATCH --account=OPEN-35-44
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=03:00:00
#SBATCH --array=0-17%3
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj3/open-35-44/verl/rl/logs/offload-factorial-%A_%a.out
#SBATCH --error=/mnt/proj3/open-35-44/verl/rl/logs/offload-factorial-%A_%a.err
set -euo pipefail
umask 007
: "${REVISION_SOURCE_ROOT:?clean immutable source checkout required}"
: "${SOURCE_COMMIT:?full frozen source commit required}"
: "${OFFLOAD_FACTORIAL_FREEZE_SHA256:?externally retained offload factorial freeze hash required}"
: "${SLURM_ARRAY_TASK_ID:?array index required}"
readonly REVISION_SOURCE_ROOT
readonly PROJECT_ROOT=/mnt/proj3/open-35-44/verl/rl
export REVISION_SOURCE_ROOT PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REVISION_SOURCE_ROOT"
export PATH="$PROJECT_ROOT/.venv-reproduce/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TZ=UTC
cd "$REVISION_SOURCE_ROOT"
exec "$PROJECT_ROOT/.venv-reproduce/bin/python" -m memory_tuner.run_offload_factorial \
  --project-root "$PROJECT_ROOT" --source-commit "$SOURCE_COMMIT" \
  --freeze-sha256 "$OFFLOAD_FACTORIAL_FREEZE_SHA256" --index "$SLURM_ARRAY_TASK_ID"
