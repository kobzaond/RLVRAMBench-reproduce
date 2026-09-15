#!/usr/bin/env bash
#SBATCH --job-name=grpo-instrumented
#SBATCH --account=OPEN-35-44
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/proj3/open-35-44/verl/rl/logs/grpo-instrumented-%j.out
#SBATCH --error=/mnt/proj3/open-35-44/verl/rl/logs/grpo-instrumented-%j.err

set -euo pipefail
umask 007

VERL_ROOT=/mnt/proj3/open-35-44/verl
REPO_ROOT=${BENCHMARK_SOURCE_ROOT:-$VERL_ROOT/rl}
IMAGE=$VERL_ROOT/images/verl-vllm018-dev1.sif
IMAGE_SHA256=3dc581ca780cd19d565445b8a02e5daa98b350b3d2116d4d64476a64a2614912
TRANSFER_QUEUE_ROOT=$REPO_ROOT/third_party/TransferQueue
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/gsm8k}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
RUN_NAME=${RUN_NAME:-grpo_instrumented/default}
PROJECT_NAME=${PROJECT_NAME:-rlvram_grpo}
OUTPUT_DIR=${BENCHMARK_OUTPUT_ROOT:-$REPO_ROOT/output}/$RUN_NAME
RUNTIME=$VERL_ROOT/runtime/$USER
SCRATCH_RUNTIME=/scratch/project/open-35-44/r/$SLURM_JOB_ID
PORT_SEED=$SLURM_JOB_ID
if [[ ${RLVRAM_GPU_UUIDS+x} ]]; then
  : "${RLVRAM_INVOCATION_ID:?paired invocation identity required}"
  : "${RLVRAM_PERIOD:?paired period required}"
  [[ "$RLVRAM_INVOCATION_ID" =~ ^[a-z0-9][a-z0-9_-]*$ ]]
  [[ "$RLVRAM_PERIOD" =~ ^[12]$ ]]
  SCRATCH_RUNTIME=$SCRATCH_RUNTIME/matched-$RLVRAM_INVOCATION_ID
  PORT_SEED=$((SLURM_JOB_ID * 3 + RLVRAM_PERIOD))
  set -o noclobber
fi
# Reserve one contiguous block for both the trainer and vLLM. vLLM otherwise
# asks the kernel for an ephemeral TCPStore port, which can race another job
# between its availability probe and bind.
VLLM_PORT_SPAN=$((8 * ${N_GPUS_PER_NODE:-2}))
MASTER_PORT_START=$(python "$REPO_ROOT/memory_tuner/select_port_block.py" \
  --seed "$PORT_SEED" --block-size "$((16 + VLLM_PORT_SPAN))")
VERL_MASTER_PORT_RANGE="$MASTER_PORT_START:$((MASTER_PORT_START + 8))"
VLLM_PORT=$((MASTER_PORT_START + 16))

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
CUSTOM_REWARD_FUNCTION_PATH=${CUSTOM_REWARD_FUNCTION_PATH:-null}
CUSTOM_REWARD_FUNCTION_NAME=${CUSTOM_REWARD_FUNCTION_NAME:-compute_score}
CAPTURE_RESPONSE_LENGTHS=${CAPTURE_RESPONSE_LENGTHS:-False}
TRAINING_SEED=${TRAINING_SEED:-42}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
ALLOCATOR_TRACE_ENABLED=${ALLOCATOR_TRACE_ENABLED:-True}
ALLOCATOR_TRACE_SYNC=${ALLOCATOR_TRACE_SYNC:-1}
RESUME_MODE=${RESUME_MODE:-auto}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-null}
DEVICE_CONTAINER_ARGS=()
DEVICE_CAPTURE_ARGS=()
DEVICE_MONITOR_ARGS=()
TRAINER_COMMAND=(python -m verl.trainer.main_ppo)
TEE_ARGS=()
CACHE_RUNTIME=$RUNTIME/cache
LOG_ROOT=$REPO_ROOT/logs
if [[ ${RLVRAM_GPU_UUIDS+x} ]]; then
  : "${RLVRAM_MODEL_SNAPSHOT:?frozen cached model snapshot required}"
  : "${RLVRAM_ALLOCATION_GPU_UUIDS:?four-GPU allocation identity required}"
  export N_GPUS_PER_NODE
  python "$REPO_ROOT/memory_tuner/device_contract.py" --validate-environment
  export RLVRAM_DEVICE_EVIDENCE_DIR=$OUTPUT_DIR/device-evidence
  DEVICE_CONTAINER_ARGS=(
    --bind "$REPO_ROOT:$REPO_ROOT"
    --env "CUDA_VISIBLE_DEVICES=$RLVRAM_GPU_UUIDS"
    --env "RLVRAM_GPU_UUIDS=$RLVRAM_GPU_UUIDS"
    --env "RLVRAM_ALLOCATION_GPU_UUIDS=$RLVRAM_ALLOCATION_GPU_UUIDS"
    --env "RLVRAM_INVOCATION_ID=$RLVRAM_INVOCATION_ID"
    --env "RLVRAM_DEVICE_EVIDENCE_DIR=$RLVRAM_DEVICE_EVIDENCE_DIR")
  DEVICE_CAPTURE_ARGS=(--resolved-model-path "$RLVRAM_MODEL_SNAPSHOT")
  DEVICE_MONITOR_ARGS=(--gpu-uuids "$RLVRAM_GPU_UUIDS"
    --device-map "$OUTPUT_DIR/gpu-device-map.json")
  TRAINER_COMMAND=(python -m memory_tuner.device_contract --train)
  TEE_ARGS=(-a)
  CACHE_RUNTIME=$SCRATCH_RUNTIME/cache
  LOG_ROOT=$OUTPUT_DIR
fi
ALLOCATOR_TRACE_DIRECTORY=
if [[ "$ALLOCATOR_TRACE_ENABLED" == "True" ]]; then
  ALLOCATOR_TRACE_DIRECTORY=$OUTPUT_DIR/allocator-traces
elif [[ "$ALLOCATOR_TRACE_ENABLED" != "False" ]]; then
  echo "ALLOCATOR_TRACE_ENABLED must be True or False" >&2
  exit 2
fi

mkdir -p \
  "$LOG_ROOT" "$OUTPUT_DIR" "$OUTPUT_DIR/allocator-traces" \
  "$RUNTIME/home" "$RUNTIME/cache/huggingface" "$RUNTIME/cache/triton" \
  "$RUNTIME/cache/torchinductor" "$RUNTIME/cache/vllm" \
  "$SCRATCH_RUNTIME/tmp" "$SCRATCH_RUNTIME/ray" \
  "$CACHE_RUNTIME/triton" "$CACHE_RUNTIME/torchinductor" "$CACHE_RUNTIME/vllm"
test -s "$DATA_DIR/train.parquet"
test -s "$DATA_DIR/test.parquet"
test -s "$IMAGE"
test -s "$TRANSFER_QUEUE_ROOT/transfer_queue/__init__.py"

MODEL_ID=$MODEL
MODEL_REQUEST=$MODEL_ID
if [[ ${RLVRAM_GPU_UUIDS+x} ]]; then
  MODEL_REQUEST=$RLVRAM_MODEL_SNAPSHOT
fi
MODEL_PATH=$(python "$REPO_ROOT/memory_tuner/resolve_hf_snapshot.py" \
  --hf-home "$RUNTIME/cache/huggingface" \
  --model "$MODEL_REQUEST")

export APPTAINER_CACHEDIR=$VERL_ROOT/cache/apptainer
export APPTAINER_TMPDIR=$SCRATCH_RUNTIME/tmp

ENVIRONMENT_JSON=$OUTPUT_DIR/environment-$SLURM_JOB_ID.json
CAPTURE_STATUS=0
python "$REPO_ROOT/memory_tuner/capture_environment.py" \
  "${DEVICE_CAPTURE_ARGS[@]}" \
  --output "$ENVIRONMENT_JSON" \
  --image "$IMAGE" \
  --image-sha256 "$IMAGE_SHA256" \
  --model "$MODEL_ID" \
  --data-dir "$DATA_DIR" \
  --repo-root "$REPO_ROOT" \
  --verl-root "$VERL_ROOT" \
  --runtime-home "$RUNTIME/home" \
  --hf-home "$RUNTIME/cache/huggingface" || CAPTURE_STATUS=$?
if (( CAPTURE_STATUS != 0 )); then
  if [[ ${RLVRAM_GPU_UUIDS+x} ]]; then
    echo "required paired environment capture failed" >&2
    exit "$CAPTURE_STATUS"
  fi
  echo "warning: environment metadata capture failed" >&2
fi

GPU_MEMORY_CSV=$OUTPUT_DIR/gpu-memory-$SLURM_JOB_ID.csv
PHASE_MEMORY_CSV=$OUTPUT_DIR/phase-memory-$SLURM_JOB_ID.csv
GPU_TELEMETRY_CSV=$OUTPUT_DIR/gpu-telemetry-$SLURM_JOB_ID.csv
PHASE_MARKER=$SCRATCH_RUNTIME/phase-marker
printf '0,0,initializing\n' > "$PHASE_MARKER"
python "$REPO_ROOT/memory_tuner/gpu_monitor.py" \
  "${DEVICE_MONITOR_ARGS[@]}" \
  --marker "$PHASE_MARKER" \
  --memory-csv "$GPU_MEMORY_CSV" \
  --phase-csv "$PHASE_MEMORY_CSV" \
  --telemetry-csv "$GPU_TELEMETRY_CSV" \
  --interval-ms "$GPU_MONITOR_INTERVAL_MS" &
GPU_MONITOR_PID=$!
FATAL_WATCHDOG_GRACE_SECONDS=${FATAL_WATCHDOG_GRACE_SECONDS:-60}
FATAL_WATCHDOG_PID=
LAUNCHER_PID=$BASHPID

cleanup() {
  if [[ -n "$FATAL_WATCHDOG_PID" ]]; then
    kill "$FATAL_WATCHDOG_PID" 2>/dev/null || true
    wait "$FATAL_WATCHDOG_PID" 2>/dev/null || true
  fi
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  rm -rf "$SCRATCH_RUNTIME"
}
trap cleanup EXIT
if [[ ${RLVRAM_GPU_UUIDS+x} ]]; then
  for _ready in {1..100}; do
    kill -0 "$GPU_MONITOR_PID"
    [[ -s "$OUTPUT_DIR/gpu-device-map.json" && -s "$GPU_MEMORY_CSV" ]] && break
    sleep 0.1
  done
  test -s "$OUTPUT_DIR/gpu-device-map.json"
  test -s "$GPU_MEMORY_CSV"
fi

RUN_LOG=$OUTPUT_DIR/training-$SLURM_JOB_ID.log
RESPONSE_LENGTH_TRACE=
if [[ "$CAPTURE_RESPONSE_LENGTHS" == "True" ]]; then
  RESPONSE_LENGTH_TRACE=$OUTPUT_DIR/response-lengths-$SLURM_JOB_ID.jsonl
fi
: > "$RUN_LOG"
fatal_engine_watchdog() {
  while ! grep -Eq \
    'CUDA out of memory\. Tried to allocate|torch\.(OutOfMemoryError|cuda\.OutOfMemoryError): CUDA out of memory|EngineCore encountered a fatal error' \
    "$RUN_LOG"; do
    sleep 5
  done
  sleep "$FATAL_WATCHDOG_GRACE_SECONDS"
  mapfile -t runtime_pids < <(
    pgrep -P "$LAUNCHER_PID" -f 'Apptainer runtime parent|apptainer exec' \
      || true
  )
  for runtime_pid in "${runtime_pids[@]}"; do
    mapfile -t payload_pids < <(
      pgrep -P "$runtime_pid" -f \
        'python .*verl\.trainer|python .*sppo_replay' || true
    )
    if (( ${#payload_pids[@]} )); then
      printf 'fatal_engine_watchdog: terminating stalled payload after %ss grace\n' \
        "$FATAL_WATCHDOG_GRACE_SECONDS" | tee -a "$RUN_LOG" >&2
      kill -TERM "${payload_pids[@]}" 2>/dev/null || true
    else
      printf 'fatal_engine_watchdog: terminating stalled runtime after %ss grace\n' \
        "$FATAL_WATCHDOG_GRACE_SECONDS" | tee -a "$RUN_LOG" >&2
      kill -TERM "$runtime_pid" 2>/dev/null || true
    fi
  done
}
fatal_engine_watchdog &
FATAL_WATCHDOG_PID=$!
STARTED_AT=$(date +%s)
set +e
apptainer exec --nv --cleanenv \
  "${DEVICE_CONTAINER_ARGS[@]}" \
  --home "$RUNTIME/home" \
  --bind "$VERL_ROOT:$VERL_ROOT" \
  --bind "/scratch/project/open-35-44:/scratch/project/open-35-44" \
  --bind "$SCRATCH_RUNTIME:/r" \
  --env PYTHONPATH="$REPO_ROOT:$TRANSFER_QUEUE_ROOT:$REPO_ROOT/instrumented_python_packages:$VERL_ROOT/python-packages:$REPO_ROOT/sppo_replay" \
  --env HF_HOME="$RUNTIME/cache/huggingface" \
  --env XDG_CACHE_HOME="$CACHE_RUNTIME" \
  --env TRITON_CACHE_DIR="$CACHE_RUNTIME/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="$CACHE_RUNTIME/torchinductor" \
  --env VLLM_CACHE_ROOT="$CACHE_RUNTIME/vllm" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TMPDIR="/r/tmp" \
  --env RAY_TMPDIR="/r/ray" \
  --env VERL_MASTER_PORT_RANGE="$VERL_MASTER_PORT_RANGE" \
  --env VLLM_PORT="$VLLM_PORT" \
  --env VLLM_PORT_SPAN="$VLLM_PORT_SPAN" \
  --env TOKENIZERS_PARALLELISM=true \
  --env NCCL_DEBUG=WARN \
  --env HYDRA_FULL_ERROR=1 \
  --env VERL_PHASE_MARKER="/r/phase-marker" \
  --env VERL_ALLOCATOR_TRACE_DIR="$ALLOCATOR_TRACE_DIRECTORY" \
  --env VERL_ALLOCATOR_TRACE_SYNC="$ALLOCATOR_TRACE_SYNC" \
  --env VERL_RESPONSE_LENGTH_TRACE="$RESPONSE_LENGTH_TRACE" \
  "$IMAGE" \
  "${TRAINER_COMMAND[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
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
    reward.custom_reward_function.path="$CUSTOM_REWARD_FUNCTION_PATH" \
    reward.custom_reward_function.name="$CUSTOM_REWARD_FUNCTION_NAME" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.seed="$TRAINING_SEED" \
    actor_rollout_ref.rollout.free_cache_engine="$FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOGPROB_MICRO_BATCH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOGPROB_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.seed="$TRAINING_SEED" \
    trainer.critic_warmup=0 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
    trainer.default_local_dir="$OUTPUT_DIR" 2>&1 | tee "${TEE_ARGS[@]}" "$RUN_LOG"
EXIT_CODE=$?
set -e

kill "$FATAL_WATCHDOG_PID" 2>/dev/null || true
wait "$FATAL_WATCHDOG_PID" 2>/dev/null || true
FATAL_WATCHDOG_PID=
ELAPSED=$(( $(date +%s) - STARTED_AT ))
kill "$GPU_MONITOR_PID" 2>/dev/null || true
MONITOR_STATUS=0
wait "$GPU_MONITOR_PID" 2>/dev/null || MONITOR_STATUS=$?
GPU_MONITOR_PID=
if [[ ${RLVRAM_GPU_UUIDS+x} ]] && (( MONITOR_STATUS != 0 )); then
  echo "paired GPU monitor failed; task evidence is invalid" >&2
  exit "$MONITOR_STATUS"
fi
PEAK_MIB=$(awk -F, '{for(i=2;i<=NF;i++)if($i+0>peak)peak=$i+0}END{print peak+0}' "$GPU_MEMORY_CSV")
printf '{"algorithm":"grpo","job_id":"%s","exit_code":%s,"elapsed_seconds":%s,"peak_gpu_memory_mib":%s,"model":"%s","gpu_count":%s,"rollout_tp_size":%s,"actor_micro_batch":%s,"rollout_logprob_micro_batch":%s,"ref_logprob_micro_batch":%s,"vllm_gpu_memory_utilization":%s,"parameter_offload":"%s","optimizer_offload":"%s","free_cache_engine":"%s","data_dir":"%s","max_prompt_length":%s,"max_response_length":%s,"max_model_len":%s,"max_num_seqs":%s,"rollout_n":%s,"train_batch_size":%s,"dataloader_num_workers":%s,"training_seed":%s,"run_log":"%s","phase_memory_csv":"%s","gpu_telemetry_csv":"%s","allocator_trace_dir":"%s","gpu_monitor_interval_ms":%s,"environment_json":"%s","response_length_trace":"%s"}\n' \
  "$SLURM_JOB_ID" "$EXIT_CODE" "$ELAPSED" "$PEAK_MIB" "$MODEL_ID" "$N_GPUS_PER_NODE" "$ROLLOUT_TP_SIZE" "$ACTOR_MICRO_BATCH" "$ROLLOUT_LOGPROB_MICRO_BATCH" "$REF_LOGPROB_MICRO_BATCH" "$VLLM_GPU_MEMORY_UTILIZATION" "$PARAMETER_OFFLOAD" "$OPTIMIZER_OFFLOAD" "$FREE_CACHE_ENGINE" "$DATA_DIR" "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$ROLLOUT_N" "$TRAIN_BATCH_SIZE" "$DATALOADER_NUM_WORKERS" "$TRAINING_SEED" "$RUN_LOG" "$PHASE_MEMORY_CSV" "$GPU_TELEMETRY_CSV" "$OUTPUT_DIR/allocator-traces" "$GPU_MONITOR_INTERVAL_MS" "$ENVIRONMENT_JSON" "$RESPONSE_LENGTH_TRACE" \
  > "$OUTPUT_DIR/trial-$SLURM_JOB_ID.json"
rm -rf "$SCRATCH_RUNTIME"
trap - EXIT
exit "$EXIT_CODE"
