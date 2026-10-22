"""Fast checks for the analytic and DeepSpeed estimator comparison."""
import json
from pathlib import Path

import pytest

from estimator_comparison import estimator
from estimator_comparison.estimator import (classify, derive_counts, fit_overhead, load_architectures, predict,
                                          stage_estimates)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmark"

BASE = {
    "model_id": "Qwen/Qwen2.5-3B-Instruct", "gpu_count": "2", "rollout_tp_size": "1",
    "actor_micro_batch": "8", "rollout_logprob_micro_batch": "4", "ref_logprob_micro_batch": "4",
    "vllm_gpu_memory_utilization": "0.6", "parameter_offload": "False", "optimizer_offload": "False",
    "free_cache_engine": "True", "max_prompt_length": "640", "max_response_length": "1024",
    "device_capacity_mib": "40960", "margin_limit_mib": "38912",
}


@pytest.fixture(scope="module")
def archs():
    return load_architectures(DATA)


def test_derived_counts_match_published_metadata():
    published = json.loads((DATA / "estimation" / "model_metadata.json").read_text())
    for model_id, fused, bias in (("Qwen/Qwen2.5-3B-Instruct", False, True),
                                  ("microsoft/Phi-4-mini-instruct", True, False),
                                  ("ibm-granite/granite-3.3-2b-instruct", False, False)):
        meta = dict(published[model_id], fused_projections=fused, attention_bias=bias)
        derived = derive_counts(meta)
        assert derived["floating_checkpoint_elements"] == meta["floating_checkpoint_elements"]
        assert derived["all_linear_layer_adapter_elements"] == meta["all_linear_layer_adapter_elements"]


def test_supplement_models_have_expected_sizes(archs):
    assert archs["Qwen/Qwen2.5-1.5B-Instruct"]["floating_checkpoint_elements"] == 1_543_714_304
    assert archs["mistralai/Mistral-7B-Instruct-v0.3"]["floating_checkpoint_elements"] == 7_248_023_552


def test_generation_stage_is_reservation_plus_resident_shard(archs):
    arch = archs[BASE["model_id"]]
    est = stage_estimates(BASE, arch, overhead_mib=0.0)
    shard = arch["floating_checkpoint_elements"] * 2 / 2 / 2 ** 20
    lora = arch["all_linear_layer_adapter_elements"] * 16 / 2 / 2 ** 20
    assert est["stages"]["generation"] == pytest.approx(0.6 * 40960 + shard + lora)
    offloaded = stage_estimates(dict(BASE, parameter_offload="True"), arch, overhead_mib=0.0)
    assert offloaded["stages"]["generation"] < est["stages"]["generation"]
    assert offloaded["stages"]["actor_update"] == pytest.approx(est["stages"]["actor_update"])


def test_monotone_in_reservation_batch_and_gpus(archs):
    arch = archs[BASE["model_id"]]
    lo = stage_estimates(BASE, arch)
    assert stage_estimates(dict(BASE, vllm_gpu_memory_utilization="0.7"), arch)["stages"]["generation"] \
        == pytest.approx(lo["stages"]["generation"] + 0.1 * 40960)
    assert stage_estimates(dict(BASE, actor_micro_batch="16"), arch)["stages"]["actor_update"] \
        > lo["stages"]["actor_update"]
    assert stage_estimates(dict(BASE, gpu_count="4"), arch)["peak_mib"] < lo["peak_mib"]


def test_retained_state_without_release(archs):
    arch = archs[BASE["model_id"]]
    released = stage_estimates(BASE, arch)
    retained = stage_estimates(dict(BASE, free_cache_engine="False"), arch)
    assert retained["stages"]["actor_update"] - released["stages"]["actor_update"] == pytest.approx(0.6 * 40960)


def test_classification_thresholds():
    assert classify(38912) == "within_margin"
    assert classify(38913) == "above_margin"
    assert classify(40960) == "above_margin"
    assert classify(40961) == "memory_failure"
    assert predict(dict(BASE, vllm_gpu_memory_utilization="0.98"),
                   load_architectures(DATA)[BASE["model_id"]])["predicted_state"] == "memory_failure"


def test_fit_overhead_uses_only_completed_sources(archs):
    arch = archs[BASE["model_id"]]
    base = stage_estimates(BASE, arch, overhead_mib=0.0)["peak_mib"]
    sources = [{"settings": BASE, "measurements": {"max_completed_run_peak_mib": str(base + 3000)}},
               {"settings": BASE, "measurements": {"max_completed_run_peak_mib": ""}}]
    value, used = fit_overhead(sources, archs)
    assert used == 1 and value == pytest.approx(3000)
    assert fit_overhead(sources[1:], archs) == (estimator.DEFAULT_OVERHEAD_MIB, 0)


def test_deepspeed_record_matches_formula_replica():
    from estimator_comparison.deepspeed_baseline import ds_zero2_gpu_bytes, ds_zero3_gpu_bytes, load_record
    record = load_record()  # raises if any recorded DeepSpeed value differs from the replica
    assert record["deepspeed_version"] == "0.19.8"
    # Qwen2.5-3B, 2 GPUs, ZeRO-3 without offload: DeepSpeed prints 27.03 GB per GPU
    gpu = ds_zero3_gpu_bytes(3_085_938_688, 311_164_928, 2, 1, cpu_offload=False, cpu_offload_params=False)
    assert gpu == 29_018_107_904 and round(gpu / 2 ** 30, 2) == 27.03
    assert ds_zero2_gpu_bytes(10, 2, 1, cpu_offload=False) == 20 + 90
    call = record["index"][("Qwen/Qwen2.5-3B-Instruct", "full", 3, 2, False, False)]
    assert call["gpu_bytes"] == gpu


def test_deepspeed_predictors_add_terms(archs):
    from estimator_comparison.deepspeed_baseline import load_record, predict as ds_predict
    record = load_record()
    arch = archs[BASE["model_id"]]
    lora = ds_predict(BASE, arch, record, "ds3_lora+frozen_base")
    with_res = ds_predict(BASE, arch, record, "ds3_lora+frozen_base+reservation")
    assert with_res - lora == pytest.approx(0.6 * 40960)
    assert ds_predict(BASE, arch, record, "ds3_full") > lora


def test_live_deepspeed_if_installed():
    stage3 = pytest.importorskip("deepspeed.runtime.zero.stage3")
    from estimator_comparison.deepspeed_baseline import ds_zero3_gpu_bytes
    for off, offp in ((False, False), (True, False), (True, True)):
        _, gpu, _ = stage3.estimate_zero3_model_states_mem_needs(123_456_789, 4_567_890, 4, 1, off, offp)
        assert gpu == ds_zero3_gpu_bytes(123_456_789, 4_567_890, 4, 1, off, offp)
