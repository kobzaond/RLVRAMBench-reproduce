#!/usr/bin/env bash
#SBATCH --job-name=rlvram-matched-gpu
#SBATCH --account=OPEN-35-44
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=00:55:00
#SBATCH --array=0-17%3
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj3/open-35-44/verl/rl/logs/matched-gpu-%A_%a.out
#SBATCH --error=/mnt/proj3/open-35-44/verl/rl/logs/matched-gpu-%A_%a.err
set -euo pipefail
umask 007
: "${REVISION_SOURCE_ROOT:?frozen source checkout required}"
: "${REVISION_PROJECT_ROOT:?project output root required}"
: "${MATRIX_PATH:?frozen matrix required}"
: "${MATRIX_SHA256:?frozen matrix digest required}"
: "${PROTOCOL_PATH:?frozen protocol required}"
: "${PROTOCOL_SHA256:?frozen protocol digest required}"
: "${SOURCE_COMMIT:?full frozen source commit required}"
export PYTHONDONTWRITEBYTECODE=1
cd "$REVISION_SOURCE_ROOT"
exec "$REVISION_PROJECT_ROOT/.venv-analysis/bin/python" -m memory_tuner.run_matched_gpu \
  --matrix "$MATRIX_PATH" --matrix-sha256 "$MATRIX_SHA256" \
  --protocol "$PROTOCOL_PATH" --protocol-sha256 "$PROTOCOL_SHA256" \
  --source-commit "$SOURCE_COMMIT" --index "$SLURM_ARRAY_TASK_ID" \
  --project-root "$REVISION_PROJECT_ROOT"
