#!/usr/bin/env bash
#SBATCH --job-name=rlvram-mi250x
#SBATCH --account=project_465003040
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465003040/rlvram_amd/logs/%x-%j.out
#SBATCH --error=/scratch/project_465003040/rlvram_amd/logs/%x-%j.err
# Submit with --gpus-per-node=$N_GPUS_PER_NODE --cpus-per-task=$((7*N)) --mem=$((60*N))G.
#
# MI250X counterpart of ../run_grpo_instrumented.slurm. The VERL overrides
# (LoRA rank/alpha, KL loss, offload switches, micro-batches, vLLM settings)
# are kept identical; only the platform layer differs: ROCm container plus a
# project venv, an amdgpu-sysfs memory sampler, and a phase-marker hook that
# replaces the A100 study's patched VERL source tree.

set -euo pipefail
umask 007

AMD_ROOT=${AMD_ROOT:-/scratch/project_465003040/rlvram_amd}
REPO_ROOT=${BENCHMARK_SOURCE_ROOT:-/project/project_465003040/RLVRAMBench-reproduce}
IMAGE=$AMD_ROOT/vllm-rocm.sif
VENV=$AMD_ROOT/venv
WORKLOAD=${WORKLOAD:-gsm8k}
DATA_DIR=${DATA_DIR:-$AMD_ROOT/data/$WORKLOAD}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
RUN_NAME=${RUN_NAME:-default}
OUTPUT_DIR=$AMD_ROOT/output/$RUN_NAME
ALGORITHM=${ALGORITHM:-grpo}              # grpo | ppo (GAE with a LoRA critic)
ADAPTATION=${ADAPTATION:-lora}            # lora | full

ACTOR_MICRO_BATCH=${ACTOR_MICRO_BATCH:-16}
ROLLOUT_LOGPROB_MICRO_BATCH=${ROLLOUT_LOGPROB_MICRO_BATCH:-8}
REF_LOGPROB_MICRO_BATCH=${REF_LOGPROB_MICRO_BATCH:-8}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.50}
PARAMETER_OFFLOAD=${PARAMETER_OFFLOAD:-False}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ROLLOUT_N=${ROLLOUT_N:-5}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-128}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-128}
GPU_MONITOR_INTERVAL_MS=${GPU_MONITOR_INTERVAL_MS:-100}
TRAINING_SEED=${TRAINING_SEED:-42}
FATAL_WATCHDOG_GRACE_SECONDS=${FATAL_WATCHDOG_GRACE_SECONDS:-60}

LORA_ARGS=(actor_rollout_ref.model.lora_rank=64 actor_rollout_ref.model.lora_alpha=32)
ACTOR_LR=3e-6
if [[ "$ADAPTATION" == "full" ]]; then
  LORA_ARGS=(actor_rollout_ref.model.lora_rank=0)
  ACTOR_LR=1e-6
elif [[ "$ADAPTATION" != "lora" ]]; then
  echo "ADAPTATION must be lora or full" >&2; exit 2
fi
ALGORITHM_ARGS=(algorithm.adv_estimator=grpo actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.rollout.n="$ROLLOUT_N")
if [[ "$ALGORITHM" == "ppo" ]]; then
  # PPO keeps the rollout count and adds a full-parameter value model
  # colocated on the same devices. (PEFT cannot wrap VERL's token-
  # classification critic as a causal-LM adapter in this stack.)
  ALGORITHM_ARGS=(algorithm.adv_estimator=gae actor_rollout_ref.actor.use_kl_loss=False
    algorithm.use_kl_in_reward=True actor_rollout_ref.rollout.n="$ROLLOUT_N"
    critic.model.path="\$MODEL_PATH" critic.model.use_remove_padding=True critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=1e-5 critic.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH"
    critic.fsdp.param_offload="$PARAMETER_OFFLOAD"
    critic.fsdp.optimizer_offload="$OPTIMIZER_OFFLOAD")
elif [[ "$ALGORITHM" != "grpo" ]]; then
  echo "ALGORITHM must be grpo or ppo" >&2; exit 2
fi

SCRATCH_RUNTIME=$AMD_ROOT/runtime/$SLURM_JOB_ID
# A requeued job keeps its job ID; never reuse a killed attempt's caches.
rm -rf "$SCRATCH_RUNTIME"
mkdir -p "$OUTPUT_DIR" "$SCRATCH_RUNTIME/tmp" "$SCRATCH_RUNTIME/ray" "$SCRATCH_RUNTIME/cache" "$AMD_ROOT/logs"
test -s "$DATA_DIR/train.parquet"
test -s "$DATA_DIR/test.parquet"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

export HF_HOME=$AMD_ROOT/hf
MODEL_DIR=$HF_HOME/hub/models--${MODEL//\//--}
MODEL_PATH=$MODEL_DIR/snapshots/$(cat "$MODEL_DIR/refs/main")
test -s "$MODEL_PATH/config.json"
ALGORITHM_ARGS=("${ALGORITHM_ARGS[@]//\$MODEL_PATH/$MODEL_PATH}")

# Private /tmp per job: VERL names its weight-transfer ZMQ socket
# ipc:///tmp/rl-colocate-zmq-<ray job id>-..., and every fresh Ray cluster
# reuses the same first job id, so co-located jobs sharing the host /tmp
# would connect to each other's sockets (hipErrorInvalidDevicePointer).
# /dev/shm stays node-wide: Singularity refuses to bind another directory
# onto /dev. Rare start-up losses of vLLM's psm_* ring buffers are recorded
# as infrastructure failures (runtime amendment 02).
SING=(env SINGULARITYENV_PYTHONNOUSERSITE=1 singularity exec -B /scratch/project_465003040 -B /project/project_465003040 -B "$SCRATCH_RUNTIME:/r" -B "$SCRATCH_RUNTIME/tmp:/tmp" "$IMAGE")

PHASE_MARKER=$SCRATCH_RUNTIME/phase-marker
printf '0,0,initializing\n' > "$PHASE_MARKER"
GPU_MEMORY_CSV=$OUTPUT_DIR/gpu-memory-$SLURM_JOB_ID.csv
PHASE_MEMORY_CSV=$OUTPUT_DIR/phase-memory-$SLURM_JOB_ID.csv
GPU_TELEMETRY_CSV=$OUTPUT_DIR/gpu-telemetry-$SLURM_JOB_ID.csv
DEVICE_MAP=$OUTPUT_DIR/gpu-device-map-$SLURM_JOB_ID.json
"${SING[@]}" python3 "$REPO_ROOT/amd_mi250x/gpu_monitor_amd.py" \
  --marker "$PHASE_MARKER" --memory-csv "$GPU_MEMORY_CSV" --phase-csv "$PHASE_MEMORY_CSV" \
  --telemetry-csv "$GPU_TELEMETRY_CSV" --device-map "$DEVICE_MAP" \
  --interval-ms "$GPU_MONITOR_INTERVAL_MS" &
GPU_MONITOR_PID=$!
for _ready in {1..300}; do
  [[ -s "$GPU_MEMORY_CSV" ]] && break
  kill -0 "$GPU_MONITOR_PID"
  sleep 0.2
done
test -s "$GPU_MEMORY_CSV"

RUN_LOG=$OUTPUT_DIR/training-$SLURM_JOB_ID.log
: > "$RUN_LOG"
LAUNCHER_PID=$BASHPID
FATAL_PATTERN='invalid device pointer|Worker failed with error|CUDA Error: out of memory|(CUDA|HIP) out of memory\. Tried to allocate|OutOfMemoryError|EngineCore encountered a fatal error|No available memory for the cache blocks|larger than the available KV cache memory|is less than desired GPU memory utilization'
fatal_engine_watchdog() {
  while ! grep -Eq "$FATAL_PATTERN" "$RUN_LOG"; do sleep 5; done
  sleep "$FATAL_WATCHDOG_GRACE_SECONDS"
  printf 'fatal_engine_watchdog: terminating stalled payload after %ss grace\n' \
    "$FATAL_WATCHDOG_GRACE_SECONDS" | tee -a "$RUN_LOG" >&2
  pkill -TERM -P "$LAUNCHER_PID" -f 'singularity exec|Singularity runtime parent' || true
}
fatal_engine_watchdog &
FATAL_WATCHDOG_PID=$!
cleanup() {
  kill "$FATAL_WATCHDOG_PID" "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  rm -rf "$SCRATCH_RUNTIME"
}
trap cleanup EXIT

# Record the exact harness sources this invocation executes. The batch
# script is Slurm's submission-time copy ($0), not the file in the checkout;
# the monitor and hook are read from the checkout at run time.
SOURCE_SHA256=$( { sha256sum "$0" | sed 's|  .*|  run_grpo_mi250x.slurm|';
  cd "$REPO_ROOT/amd_mi250x" && sha256sum gpu_monitor_amd.py hook/sitecustomize.py; } \
  | awk '{printf "%s%s:%s", (NR>1?";":""), $2, $1}')
STARTED_AT=$(date +%s)
set +e
"${SING[@]}" bash -c "
# The Slurm device cgroup already restricts the container to the allocated
# GCDs. Ray then assigns each worker its device through HIP_VISIBLE_DEVICES;
# an inherited CUDA_VISIBLE_DEVICES list would conflict with it in vLLM.
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
source '$VENV/bin/activate'
export PYTHONNOUSERSITE=1 PYTHONPATH='$REPO_ROOT/amd_mi250x/hook'
export VERL_PHASE_MARKER=/r/phase-marker
export HF_HOME='$HF_HOME' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR=/r/tmp RAY_TMPDIR=/r/ray XDG_CACHE_HOME=/r/cache
export TRITON_CACHE_DIR=/r/cache/triton VLLM_CACHE_ROOT=/r/cache/vllm TORCHINDUCTOR_CACHE_DIR=/r/cache/inductor
export TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
# VLLM_PORT is deliberately unset: each job runs one vLLM server per GPU, and
# a shared base port makes those servers race each other (runtime amendment
# 03). Rare ephemeral-port races with co-located jobs remain possible.
exec python3 -m verl.trainer.main_ppo \"\$@\"
" bash \
    trainer.use_v1=False \
    "${ALGORITHM_ARGS[@]}" \
    trainer.val_before_train=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_max_samples="$TRAIN_MAX_SAMPLES" \
    data.val_max_samples="$VAL_MAX_SAMPLES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.seed="$TRAINING_SEED" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    "${LORA_ARGS[@]}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="$PARAMETER_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.seed="$TRAINING_SEED" \
    actor_rollout_ref.actor.data_loader_seed="$TRAINING_SEED" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.seed="$TRAINING_SEED" \
    actor_rollout_ref.rollout.free_cache_engine="$FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon="${LAYERED_SUMMON:-True}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOGPROB_MICRO_BATCH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOGPROB_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.seed="$TRAINING_SEED" \
    trainer.critic_warmup=0 \
    ${EXTRA_OVERRIDES:-} \
    trainer.logger='["console"]' \
    trainer.project_name=rlvram_mi250x \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
    hydra.run.dir="$OUTPUT_DIR/hydra-$SLURM_JOB_ID" 2>&1 | tee "$RUN_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e
ELAPSED=$(( $(date +%s) - STARTED_AT ))
kill "$FATAL_WATCHDOG_PID" 2>/dev/null || true
kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
PEAK_MIB=$(awk -F, '{for(i=2;i<=NF;i++)if($i+0>peak)peak=$i+0}END{print peak+0}' "$GPU_MEMORY_CSV")
cat > "$OUTPUT_DIR/trial-$SLURM_JOB_ID.json" <<JSON
{"accelerator":"AMD Instinct MI250X (one GCD per device)","algorithm":"$ALGORITHM","adaptation":"$ADAPTATION","job_id":"$SLURM_JOB_ID","node":"$(hostname)","exit_code":$EXIT_CODE,"elapsed_seconds":$ELAPSED,"peak_gpu_memory_mib":$PEAK_MIB,"model":"$MODEL","model_path":"$MODEL_PATH","workload":"$WORKLOAD","gpu_count":$N_GPUS_PER_NODE,"rollout_tp_size":$ROLLOUT_TP_SIZE,"actor_micro_batch":$ACTOR_MICRO_BATCH,"rollout_logprob_micro_batch":$ROLLOUT_LOGPROB_MICRO_BATCH,"ref_logprob_micro_batch":$REF_LOGPROB_MICRO_BATCH,"vllm_gpu_memory_utilization":$VLLM_GPU_MEMORY_UTILIZATION,"parameter_offload":"$PARAMETER_OFFLOAD","optimizer_offload":"$OPTIMIZER_OFFLOAD","free_cache_engine":"$FREE_CACHE_ENGINE","max_prompt_length":$MAX_PROMPT_LENGTH,"max_response_length":$MAX_RESPONSE_LENGTH,"max_model_len":$MAX_MODEL_LEN,"max_num_seqs":$MAX_NUM_SEQS,"rollout_n":$ROLLOUT_N,"train_batch_size":$TRAIN_BATCH_SIZE,"total_training_steps":$TOTAL_TRAINING_STEPS,"training_seed":$TRAINING_SEED,"gpu_monitor_interval_ms":$GPU_MONITOR_INTERVAL_MS,"harness_sha256":"$SOURCE_SHA256","image_sha256":"$(cut -d' ' -f1 "$AMD_ROOT/vllm-rocm.sif.sha256")","run_log":"$RUN_LOG","phase_memory_csv":"$PHASE_MEMORY_CSV","device_map":"$DEVICE_MAP"}
JSON
exit "$EXIT_CODE"
