#!/usr/bin/env bash
#SBATCH --job-name=llmem-native-dev
#SBATCH --account=OPEN-35-44
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj3/open-35-44/verl/rl/logs/llmem-%j.out
#SBATCH --error=/mnt/proj3/open-35-44/verl/rl/logs/llmem-%j.err
set -euo pipefail
trap 'printf "LLMem development setup failed at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR
module load Python/3.10.8-GCCcore-12.2.0-bare CUDA/11.7.0
export CC=/apps/all/GCCcore/11.3.0/bin/gcc
export CXX=/apps/all/GCCcore/11.3.0/bin/g++
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=/home/kobzaond/rlvrambench-external/torch-extensions
: "${REVISION_SOURCE_ROOT:?source checkout required}"
export PYTHONPATH="$REVISION_SOURCE_ROOT"
cd "$REVISION_SOURCE_ROOT"
native_python=/home/kobzaond/rlvrambench-llmem-env/bin/python
native_model=/home/kobzaond/rlvrambench-external/models/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6
# NVML indices can be remapped by Slurm device isolation. This allocation
# requests exactly one GPU; require one visible UUID instead of a host index.
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
native_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ "$native_uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]]
export CUDA_VISIBLE_DEVICES="$native_uuid"
for native_mode in estimate measure; do
  "$native_python" -m memory_tuner.llmem_native \
    --mode "$native_mode" --model "$native_model" --batch 1 --sequence 512 --seed 431 \
    --gpu-uuid "$native_uuid" --port "$((20000 + SLURM_JOB_ID % 20000))" \
    --output "/mnt/proj3/open-35-44/verl/rl/output/llmem_native_development/j${SLURM_JOB_ID}/${native_mode}"
done
