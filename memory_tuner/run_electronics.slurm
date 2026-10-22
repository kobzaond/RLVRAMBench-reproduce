#!/usr/bin/env bash
#SBATCH --job-name=rlvram-electronics
#SBATCH --account=OPEN-35-44
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=00:55:00
#SBATCH --array=0-2%3
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj3/open-35-44/verl/rl/logs/electronics-%A_%a.out
#SBATCH --error=/mnt/proj3/open-35-44/verl/rl/logs/electronics-%A_%a.err
set -euo pipefail
: "${REVISION_SOURCE_ROOT:?immutable execution checkout required}"
: "${SOURCE_COMMIT:?frozen commit required}"
: "${PROTOCOL_SHA256:?frozen protocol required}"
: "${ELECTRONICS_PANEL:?panel name required}"
export TZ=UTC PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REVISION_SOURCE_ROOT"
export PATH="/mnt/proj3/open-35-44/verl/rl/.venv-reproduce/bin:$PATH"
cd "$REVISION_SOURCE_ROOT"
exec python -m memory_tuner.electronics_experiment run \
  --panel "$ELECTRONICS_PANEL" --source-commit "$SOURCE_COMMIT" \
  --protocol-sha256 "$PROTOCOL_SHA256" --index "$SLURM_ARRAY_TASK_ID"
