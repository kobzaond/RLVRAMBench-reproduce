#!/usr/bin/env bash
#SBATCH --job-name=llmem-native
#SBATCH --account=OPEN-35-44
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --array=0-7%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj3/open-35-44/verl/rl/logs/llmem-validation-%A_%a.out
#SBATCH --error=/mnt/proj3/open-35-44/verl/rl/logs/llmem-validation-%A_%a.err
set -euo pipefail
module load Python/3.10.8-GCCcore-12.2.0-bare CUDA/11.7.0
export CC=/apps/all/GCCcore/11.3.0/bin/gcc
export CXX=/apps/all/GCCcore/11.3.0/bin/g++
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=/home/kobzaond/rlvrambench-external/torch-extensions
: "${REVISION_SOURCE_ROOT:?immutable source required}"
: "${SOURCE_COMMIT:?source commit required}"
: "${PROTOCOL_SHA256:?protocol hash required}"
export PYTHONPATH="$REVISION_SOURCE_ROOT"
cd "$REVISION_SOURCE_ROOT"
exec /home/kobzaond/rlvrambench-llmem-env/bin/python -m memory_tuner.llmem_native_protocol run \
  --source-commit "$SOURCE_COMMIT" --protocol-sha256 "$PROTOCOL_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID"
