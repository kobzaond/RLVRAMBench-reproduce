"""Static first-principles per-device memory estimator for colocated LoRA-GRPO.

The structure is *inspired by* analytical estimators such as LLMem, DNNMem and
the static memory model in ReaL: per-device memory is the sum of explicitly
modelled tensors (weights, adapter states, activations, logits) plus a
framework overhead constant. It is **not** an implementation of any of those
tools and makes no claim to reproduce their numbers.

Only configuration settings and architecture metadata enter a prediction; no
measured outcome is used, except by the explicitly separate calibrated
variant, which fits one additive overhead constant from *source*
measurements of a benchmark task (permitted inputs).

Stages modelled per device (MiB = 2**20 bytes):

* generation: vLLM preallocates ``gpu_memory_utilization x capacity``; the
  colocated FSDP actor shard (weights + LoRA states) stays resident unless
  parameters are offloaded.
* actor_update: FSDP weight shard + LoRA params/grads/Adam (fp32, sharded) +
  transient all-gather of the largest unit + gradient-checkpointed
  activations (micro_batch x seq_len x hidden x layers x 2 bytes) + fp32
  logits (micro_batch x seq_len x vocab x 4 bytes); reference policy is the
  adapter-disabled actor (no separate resident copy, i.e. "offloaded").
* logprob (old/ref log-probabilities): weight shard + LoRA + all-gather +
  fp32 logits at the log-probability micro-batch.
* weight_sync: the vLLM engine is woken (weights and KV cache, i.e. the full
  reservation) before the actor weights are pushed, so the stage holds the
  reservation + the reloaded actor shard + LoRA + one gathered full tensor.

When ``free_cache_engine`` is False the generation reservation is retained
through the training stages (retained-state term).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from estimator_comparison.common import DEVICE_CAPACITY_MIB, MARGIN_LIMIT_MIB, to_bool, to_float

MIB = float(2 ** 20)
DEFAULT_OVERHEAD_MIB = 1024.0  # CUDA context + NCCL/communication buffers; a priori, not fitted
LORA_RANK = 64
LORA_ALPHA = 32  # scaling only; does not change memory
BF16 = 2
FP32 = 4
PACKAGE_DIR = Path(__file__).resolve().parent


def derive_counts(arch: Mapping, rank: int = LORA_RANK) -> dict:
    """Analytic parameter / all-linear LoRA element counts from config fields."""
    h = int(arch["hidden_size"])
    inter = int(arch["intermediate_size"])
    heads = int(arch["num_attention_heads"])
    kv = int(arch["num_key_value_heads"])
    hd = int(arch.get("head_dim") or h // heads)
    layers = int(arch["num_hidden_layers"])
    vocab = int(arch["vocab_size"])
    bias = bool(arch.get("attention_bias", False))
    fused = bool(arch.get("fused_projections", False))
    q_out, kv_out = heads * hd, kv * hd
    per_layer = (h * q_out + 2 * h * kv_out + q_out * h + 3 * h * inter + 2 * h)
    if bias:
        per_layer += q_out + 2 * kv_out
    embed = vocab * h
    total = layers * per_layer + embed + h + (0 if arch.get("tie_word_embeddings") else embed)
    if fused:
        adapter = rank * ((h + q_out + 2 * kv_out) + (q_out + h) + (h + 2 * inter) + (inter + h))
    else:
        adapter = rank * ((h + q_out) + 2 * (h + kv_out) + (q_out + h) + 2 * (h + inter) + (inter + h))
    return {
        "floating_checkpoint_elements": total,
        "all_linear_layer_adapter_elements": adapter * layers,
        "largest_transformer_layer_elements": per_layer,
        "largest_weight_tensor_elements": max(embed, h * inter),
    }


def load_architectures(data_dir: Path | str | None = None) -> dict:
    """model_id -> architecture dict (published metadata + documented supplement)."""
    archs: dict = {}
    supplement = json.loads((PACKAGE_DIR / "architecture_supplement.json").read_text())
    for model_id, arch in supplement.items():
        if model_id.startswith("_"):
            continue
        merged = dict(arch)
        merged.update(derive_counts(arch))
        merged["metadata_source"] = "estimator_comparison/architecture_supplement.json (config.json + analytic counts)"
        archs[model_id] = merged
    if data_dir is not None:
        path = Path(data_dir) / "estimation" / "model_metadata.json"
        if path.is_file():
            for model_id, arch in json.loads(path.read_text()).items():
                merged = dict(arch)
                merged["metadata_source"] = "benchmark/estimation/model_metadata.json"
                archs[model_id] = merged
    return archs


def _setting(settings: Mapping, key: str, default=None):
    value = settings.get(key, default)
    return default if value in (None, "") else value


def stage_estimates(settings: Mapping, arch: Mapping, overhead_mib: float = DEFAULT_OVERHEAD_MIB,
                    param_bytes: int = BF16) -> dict:
    """Return per-stage per-device memory estimates (MiB) and their components."""
    capacity = to_float(_setting(settings, "device_capacity_mib", DEVICE_CAPACITY_MIB))
    gpus = int(float(_setting(settings, "gpu_count", 1)))
    tp = int(float(_setting(settings, "rollout_tp_size", 1)))
    util = to_float(_setting(settings, "vllm_gpu_memory_utilization"))
    actor_mb = int(float(_setting(settings, "actor_micro_batch")))
    lp_mb = max(int(float(_setting(settings, "rollout_logprob_micro_batch", actor_mb))),
                int(float(_setting(settings, "ref_logprob_micro_batch", actor_mb))))
    param_offload = bool(to_bool(_setting(settings, "parameter_offload", "False")))
    optim_offload = bool(to_bool(_setting(settings, "optimizer_offload", "False")))
    free_cache = to_bool(_setting(settings, "free_cache_engine", "True"))
    free_cache = True if free_cache is None else free_cache
    prompt = to_float(_setting(settings, "max_prompt_length"))
    response = to_float(_setting(settings, "max_response_length"))
    seq = (prompt + response) if prompt is not None and response is not None else to_float(
        _setting(settings, "max_model_len"))

    params = float(arch["floating_checkpoint_elements"])
    adapter = float(arch["all_linear_layer_adapter_elements"])
    hidden = float(arch["hidden_size"])
    layers = float(arch["num_hidden_layers"])
    vocab = float(arch["vocab_size"])
    unit = max(float(arch["largest_transformer_layer_elements"]), float(arch["largest_weight_tensor_elements"]))

    weight_shard = params * param_bytes / gpus / MIB
    lora_params_grads = adapter * (FP32 + FP32) / gpus / MIB
    lora_adam = 0.0 if optim_offload else adapter * 2 * FP32 / gpus / MIB
    gathered = 2.0 * unit * BF16 / MIB  # current + prefetched FSDP unit, bf16
    reservation = util * capacity
    vllm_weights = params * BF16 / tp / MIB
    activations = actor_mb * seq * hidden * layers * BF16 / MIB
    logits = actor_mb * seq * vocab * FP32 / MIB
    lp_logits = lp_mb * seq * vocab * FP32 / MIB
    retained = 0.0 if free_cache else reservation
    largest_tensor = float(arch["largest_weight_tensor_elements"]) * BF16 / MIB

    gen_resident = (0.0 if param_offload else weight_shard + lora_params_grads) + lora_adam
    train_resident = weight_shard + lora_params_grads + lora_adam  # offloaded params are reloaded
    stages = {
        "generation": reservation + gen_resident + overhead_mib,
        "actor_update": train_resident + gathered + activations + logits + retained + overhead_mib,
        "logprob": train_resident + gathered + lp_logits + retained + overhead_mib,
        "weight_sync": reservation + train_resident + largest_tensor + overhead_mib,
    }
    peak_stage = max(stages, key=lambda k: stages[k])
    return {
        "stages": stages,
        "peak_mib": stages[peak_stage],
        "peak_stage": peak_stage,
        "components": {
            "reservation_mib": reservation,
            "weight_shard_mib": weight_shard,
            "lora_params_grads_mib": lora_params_grads,
            "lora_adam_mib": lora_adam,
            "allgather_mib": gathered,
            "activations_mib": activations,
            "logits_mib": logits,
            "logprob_logits_mib": lp_logits,
            "retained_state_mib": retained,
            "vllm_weights_mib": vllm_weights,
            "overhead_mib": overhead_mib,
        },
        "capacity_mib": capacity,
    }


def classify(peak_mib: float, margin_mib: float = MARGIN_LIMIT_MIB,
             capacity_mib: float = DEVICE_CAPACITY_MIB) -> str:
    if peak_mib <= margin_mib:
        return "within_margin"
    if peak_mib > capacity_mib:
        return "memory_failure"
    return "above_margin"


def predict(settings: Mapping, arch: Mapping, overhead_mib: float = DEFAULT_OVERHEAD_MIB,
            param_bytes: int = BF16) -> dict:
    estimate = stage_estimates(settings, arch, overhead_mib=overhead_mib, param_bytes=param_bytes)
    margin = to_float(_setting(settings, "margin_limit_mib", MARGIN_LIMIT_MIB))
    estimate["predicted_state"] = classify(estimate["peak_mib"], margin, estimate["capacity_mib"])
    return estimate


def fit_overhead(sources, archs: Mapping, param_bytes: int = BF16,
                 fallback: float = DEFAULT_OVERHEAD_MIB) -> tuple[float, int]:
    """Fit one additive constant from source measurements only.

    ``sources`` is the list of source objects of a single task (``settings``
    and ``measurements``). The constant is the mean of
    (measured max completed peak - overhead-free prediction) over sources
    with a completed peak; sources without one (e.g. memory failures) carry no
    peak and are skipped. Returns (constant, number of sources used).
    """
    residuals = []
    for source in sources:
        measured = to_float(source["measurements"].get("max_completed_run_peak_mib"))
        if measured is None:
            continue
        arch = archs[source["settings"]["model_id"]]
        base = stage_estimates(source["settings"], arch, overhead_mib=0.0, param_bytes=param_bytes)
        residuals.append(measured - base["peak_mib"])
    if not residuals:
        return fallback, 0
    return sum(residuals) / len(residuals), len(residuals)
