#!/usr/bin/env bash
#SBATCH --job-name=rlvram-device-preflight
#SBATCH --account=OPEN-35-44
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:12:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/proj3/open-35-44/verl/rl/logs/matched-preflight-%j.out
#SBATCH --error=/mnt/proj3/open-35-44/verl/rl/logs/matched-preflight-%j.err
set -euo pipefail
umask 007
: "${REVISION_SOURCE_ROOT:?immutable source required}"
: "${REVISION_PROJECT_ROOT:?project root required}"
export PYTHONDONTWRITEBYTECODE=1
cd "$REVISION_SOURCE_ROOT"
PROJECT_VERL_ROOT=/mnt/proj3/open-35-44/verl
PREFLIGHT_RECORDS=$REVISION_PROJECT_ROOT/output/estimation-device-preflight/$SLURM_JOB_ID
PREFLIGHT_RUNTIME=/scratch/project/open-35-44/r/$SLURM_JOB_ID/preflight
PREFLIGHT_PYTHONPATH=$REVISION_SOURCE_ROOT:$REVISION_SOURCE_ROOT/third_party/TransferQueue:$REVISION_SOURCE_ROOT/instrumented_python_packages:$PROJECT_VERL_ROOT/python-packages:$REVISION_SOURCE_ROOT/sppo_replay
mkdir -p "$PREFLIGHT_RECORDS" "$PREFLIGHT_RUNTIME/tmp"
PREFLIGHT_UUIDS=$("$REVISION_PROJECT_ROOT/.venv-analysis/bin/python" -c \
  'from memory_tuner.run_matched_gpu import allocation_context; print(",".join(g["uuid"] for g in allocation_context()["gpus"]))')
export APPTAINER_CACHEDIR=$PROJECT_VERL_ROOT/cache/apptainer
export APPTAINER_TMPDIR=$PREFLIGHT_RUNTIME/tmp
# Reuse the payload's exact host-environment sanitizer, including removal of
# site SINGULARITYENV_/APPTAINERENV_ overrides. Explicit container flags below
# carry the selected device contract back in; no numeric CUDA mask is inferred.
exec "$REVISION_PROJECT_ROOT/.venv-analysis/bin/python" -c \
  'import os, sys; from memory_tuner.run_matched_gpu import clean_environment; os.execvpe(sys.argv[1], sys.argv[1:], clean_environment({}))' \
  apptainer exec --nv --cleanenv \
  --home "$PROJECT_VERL_ROOT/runtime/$USER/home" \
  --bind "$PROJECT_VERL_ROOT:$PROJECT_VERL_ROOT" \
  --bind "$REVISION_SOURCE_ROOT:$REVISION_SOURCE_ROOT" \
  --bind "/scratch/project/open-35-44:/scratch/project/open-35-44" \
  --bind "$PREFLIGHT_RUNTIME:/r" \
  --env "PYTHONPATH=$PREFLIGHT_PYTHONPATH" \
  --env "HF_HOME=$PROJECT_VERL_ROOT/runtime/$USER/cache/huggingface" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "HF_DATASETS_OFFLINE=1" \
  --env "XDG_CACHE_HOME=/r/cache" \
  --env "TRITON_CACHE_DIR=/r/cache/triton" \
  --env "TORCHINDUCTOR_CACHE_DIR=/r/cache/torchinductor" \
  --env "VLLM_CACHE_ROOT=/r/cache/vllm" \
  --env "TOKENIZERS_PARALLELISM=true" \
  --env "NCCL_DEBUG=WARN" \
  --env "HYDRA_FULL_ERROR=1" \
  --env "CUDA_VISIBLE_DEVICES=$PREFLIGHT_UUIDS" \
  --env "RLVRAM_GPU_UUIDS=$PREFLIGHT_UUIDS" \
  --env "RLVRAM_ALLOCATION_GPU_UUIDS=$PREFLIGHT_UUIDS" \
  --env "RLVRAM_INVOCATION_ID=preflight-$SLURM_JOB_ID" \
  --env "RLVRAM_DEVICE_EVIDENCE_DIR=$PREFLIGHT_RECORDS" \
  --env "TMPDIR=/r/tmp" \
  --env "RLVRAM_PREFLIGHT_RAY_ROOT=/r/ray" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  "$PROJECT_VERL_ROOT/images/verl-vllm018-dev1.sif" \
  python -m memory_tuner.device_contract --preflight
