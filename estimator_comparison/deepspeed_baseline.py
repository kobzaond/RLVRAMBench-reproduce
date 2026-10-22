"""DeepSpeed's published ZeRO memory estimators as an external baseline.

DeepSpeed ships analytic model-state estimators
(``deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs`` and
``deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs``,
wrapped by the ``*_all_cold`` reporting functions). They were **called for
real** in a separate environment by ``deepspeed_collect.py``; the returned
values, versions and source hashes are stored in ``deepspeed_estimates.json``.
This module only reads that record (so the main analysis needs no torch);
``ds_zero3_gpu_bytes``/``ds_zero2_gpu_bytes`` replicate the formulas only to
verify the record and for tests.

Mapping of a benchmark configuration to an estimator call (ZeRO-3 ~ FSDP
full sharding, which VERL uses):
* ``parameter_offload=True``  -> cpu_offload=True, cpu_offload_params=True
* ``optimizer_offload=True``  -> cpu_offload=True, cpu_offload_params=False
* otherwise                   -> no offload
* num_gpus_per_node = gpu_count, num_nodes = 1, zero_init=True.

Trainable-parameter assumptions:
* ``full``: every checkpoint parameter trainable (the estimator's own
  assumption);
* ``lora``: only the rank-64 all-linear adapters are trainable; the frozen
  bf16 base is added explicitly (2 bytes x params / GPUs resident, or one
  bf16 largest layer when parameters are offloaded).

Predictors (all from target settings only): DS model states alone; + vLLM
reservation (u x capacity); + reservation + the analytic activation/logit term.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from estimator_comparison.common import MODEL_LABELS, STATES, Benchmark, to_bool, to_float, write_csv, write_json
from estimator_comparison.estimator import classify, load_architectures, stage_estimates
from estimator_comparison.estimator_eval import KEY_METRICS, REFERENCE_RULES, TRACKS, _bench_cli, _err_stats

MIB = float(2 ** 20)
RECORD = Path(__file__).resolve().parent / "deepspeed_estimates.json"
PREDICTORS = {
    "ds3_full": ("zero3", "full", False, False),
    "ds3_full+reservation": ("zero3", "full", True, False),
    "ds3_full+reservation+activations": ("zero3", "full", True, True),
    "ds3_lora+frozen_base": ("zero3", "lora", False, False),
    "ds3_lora+frozen_base+reservation": ("zero3", "lora", True, False),
    "ds3_lora+frozen_base+reservation+activations": ("zero3", "lora", True, True),
    "ds2_full": ("zero2", "full", False, False),
    "ds2_full+reservation": ("zero2", "full", True, False),
}


# --- exact replicas of DeepSpeed 0.19.8 formulas (GPU part only), for verification ---------
def ds_zero3_gpu_bytes(total_params, largest_layer_params, num_gpus_per_node=1, num_nodes=1,
                       cpu_offload=True, cpu_offload_params=True):
    """deepspeed/runtime/zero/stage3.py::estimate_zero3_model_states_mem_needs (gpu_mem)."""
    total_gpus = num_nodes * num_gpus_per_node
    largest_layer_memory = 4 * largest_layer_params
    if cpu_offload:
        if cpu_offload_params:
            return int(largest_layer_memory)
        return int(largest_layer_memory + int(2 * total_params / total_gpus))
    return int(largest_layer_memory + int(18 * total_params / total_gpus))


def ds_zero2_gpu_bytes(total_params, num_gpus_per_node=1, num_nodes=1, cpu_offload=True):
    """deepspeed/runtime/zero/stage_1_and_2.py::estimate_zero2_model_states_mem_needs (gpu_mem)."""
    total_gpus = num_nodes * num_gpus_per_node
    if cpu_offload:
        return int(2 * total_params)
    return int(2 * total_params + int(18 * total_params / total_gpus))


def load_record(path: Path = RECORD) -> dict:
    record = json.loads(path.read_text())
    index = {}
    for call in record["calls"]:
        key = (call["model_id"], call["trainable"], call["stage"], call["num_gpus_per_node"],
               call["cpu_offload"], call["cpu_offload_params"])
        index[key] = call
        if call["stage"] == 3:
            replica = ds_zero3_gpu_bytes(call["total_params"], call["largest_layer_params"],
                                         call["num_gpus_per_node"], 1, call["cpu_offload"], call["cpu_offload_params"])
        else:
            replica = ds_zero2_gpu_bytes(call["total_params"], call["num_gpus_per_node"], 1, call["cpu_offload"])
        if replica != call["gpu_bytes"]:
            raise ValueError(f"Recorded DeepSpeed value differs from formula replica: {key}")
    record["index"] = index
    return record


def model_state_mib(settings, arch, record, stage, trainable):
    gpus = int(float(settings["gpu_count"]))
    p_off = bool(to_bool(settings.get("parameter_offload", "False")))
    o_off = bool(to_bool(settings.get("optimizer_offload", "False")))
    if stage == "zero3":
        offload, offload_params = (True, True) if p_off else ((True, False) if o_off else (False, False))
        call = record["index"][(settings["model_id"], trainable, 3, gpus, offload, offload_params)]
    else:
        call = record["index"][(settings["model_id"], trainable, 2, gpus, o_off, None)]
    value = call["gpu_bytes"] / MIB
    if trainable == "lora":
        if p_off:
            value += 2 * max(arch["largest_transformer_layer_elements"], arch["largest_weight_tensor_elements"]) / MIB
        else:
            value += 2 * arch["floating_checkpoint_elements"] / gpus / MIB
    return value


def predict(settings, arch, record, predictor):
    stage, trainable, add_res, add_act = PREDICTORS[predictor]
    value = model_state_mib(settings, arch, record, stage, trainable)
    comp = stage_estimates(settings, arch, overhead_mib=0.0)["components"]
    if add_res:
        value += comp["reservation_mib"]
    if add_act:
        value += comp["activations_mib"] + comp["logits_mib"]
    return value


def run(bench: Benchmark, out, repo_root, configs) -> dict:
    record = load_record()
    archs = load_architectures(bench.root)
    data = bench.root.resolve()
    inputs_dir, pred_dir, score_dir = out / "task_inputs", out / "deepspeed_predictions", out / "deepspeed_scores"
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    all_rows = defaultdict(list)
    for track in TRACKS:
        path = inputs_dir / f"{track}.json"
        if not path.is_file():
            _bench_cli(repo_root, "inputs", "--data", data, "--track", track, "--output", path)
        tasks = json.loads(path.read_text())
        for name in PREDICTORS:
            rows = []
            for task in tasks:
                for target in task["targets"]:  # target settings only
                    s = target["settings"]
                    peak = predict(s, archs[s["model_id"]], record, name)
                    rows.append({"query_id": target["query_id"], "predicted_state": classify(peak),
                                 "track": track, "target_configuration_id": s["configuration_id"],
                                 "predicted_peak_mib": peak})
            rows.sort(key=lambda r: r["query_id"])
            all_rows[name].extend(rows)
            fname = name.replace("+", "_plus_")
            write_csv(pred_dir / f"{fname}_{track}.csv", rows, fieldnames=["query_id", "predicted_state"])
            _bench_cli(repo_root, "evaluate", "--data", data, "--track", track,
                       "--predictions", pred_dir / f"{fname}_{track}.csv", "--output", score_dir / f"{fname}_{track}.json")
    for name, rows in all_rows.items():
        fname = name.replace("+", "_plus_")
        rows.sort(key=lambda r: r["query_id"])
        write_csv(pred_dir / f"{fname}_all.csv", rows, fieldnames=["query_id", "predicted_state"])
        _bench_cli(repo_root, "evaluate", "--data", data, "--track", "all",
                   "--predictions", pred_dir / f"{fname}_all.csv", "--output", score_dir / f"{fname}_all.json")

    reference = json.loads((bench.root / "reference_scores.json").read_text())
    comparison = []
    for track in TRACKS + ("all",):
        for name in PREDICTORS:
            block = json.loads((score_dir / f"{name.replace('+', '_plus_')}_{track}.json").read_text())["overall"]
            comparison.append({"track": track, "method": name, **{k: block.get(k) for k in KEY_METRICS}})
        for variant in ("analytic_bf16", "analytic_fp32_master", "calibrated_bf16"):
            path = out / "analytic_scores" / f"{variant}_{track}.json"
            if path.is_file():
                block = json.loads(path.read_text())["overall"]
                comparison.append({"track": track, "method": variant, **{k: block.get(k) for k in KEY_METRICS}})
        for rule in REFERENCE_RULES:
            block = reference[rule]["overall"] if track == "all" else reference[rule]["per_track"][track]
            comparison.append({"track": track, "method": rule, **{k: block.get(k) for k in KEY_METRICS}})
    write_csv(out / "score_comparison_all_methods.csv", comparison)

    # peak error on completed configurations (all panels with architecture metadata)
    err_rows = []
    for c in configs:
        s = c["settings"]
        if s["model_id"] not in archs or c["max_completed_peak_mib"] is None or c["state"] == "memory_failure" \
                or c["completed_processes"] == 0 or (c["state"] == "unresolved" and c["source"] != "replication_control"):
            continue
        rec = {"source": c["source"], "configuration_id": c["configuration_id"], "model_family": s["model_family"],
               "gpu_count": s["gpu_count"], "vllm_gpu_memory_utilization": s["vllm_gpu_memory_utilization"],
               "parameter_offload": s["parameter_offload"], "free_cache_engine": s["free_cache_engine"],
               "observed_state": c["state"], "measured_peak_mib": c["max_completed_peak_mib"],
               "measured_actor_update_mib": c.get("actor_update_peak_mean_mib")}
        for name in PREDICTORS:
            p = predict(s, archs[s["model_id"]], record, name)
            rec[f"{name}_mib"] = p
            rec[f"{name}_error_mib"] = p - c["max_completed_peak_mib"]
            rec[f"{name}_state"] = classify(p)
        err_rows.append(rec)
    write_csv(out / "deepspeed_peak_error_configurations.csv", err_rows)

    fail_rows = [c for c in configs if c["state"] == "memory_failure" and c["settings"]["model_id"] in archs]
    peak_errors = {}
    for name in PREDICTORS:
        errs = [r[f"{name}_error_mib"] for r in err_rows]
        agree = sum(r[f"{name}_state"] == r["observed_state"] for r in err_rows if r["observed_state"] in STATES)
        actor = [r["ds3_lora+frozen_base_mib"] - r["measured_actor_update_mib"] for r in err_rows
                 if r["measured_actor_update_mib"] is not None] if name == "ds3_lora+frozen_base" else None
        peak_errors[name] = {
            "whole_run_peak": _err_stats(errs),
            "completed_label_agreement": f"{agree}/{sum(r['observed_state'] in STATES for r in err_rows)}",
            "failing_configs_predicted": {st: sum(classify(predict(c["settings"], archs[c["settings"]["model_id"]],
                                                                   record, name)) == st for c in fail_rows)
                                          for st in STATES},
            **({"vs_measured_actor_update_stage": _err_stats(actor)} if actor else {}),
        }
    # share of the measured peak explained by each term (completed configs)
    shares = {
        "median_measured_peak_mib": median(r["measured_peak_mib"] for r in err_rows),
        "median_ds3_lora_model_states_mib": median(r["ds3_lora+frozen_base_mib"] for r in err_rows),
        "median_ds3_full_model_states_mib": median(r["ds3_full_mib"] for r in err_rows),
        "median_reservation_mib": median(float(r["vllm_gpu_memory_utilization"]) * 40960 for r in err_rows),
        "median_fraction_of_peak_lora_model_states": median(r["ds3_lora+frozen_base_mib"] / r["measured_peak_mib"]
                                                            for r in err_rows),
        "median_fraction_of_peak_reservation": median(float(r["vllm_gpu_memory_utilization"]) * 40960 / r["measured_peak_mib"]
                                                      for r in err_rows),
    }
    summary = {
        "deepspeed_version": record["deepspeed_version"], "torch_version": record["torch_version"],
        "estimator_sources": {k: {"file": v["file"], "sha256": v["sha256"],
                                  "functions": {f: x["first_line"] for f, x in v["functions"].items()}}
                              for k, v in record["sources"].items()},
        "called_for_real": True,
        "completed_configurations": len(err_rows), "failing_configurations": len(fail_rows),
        "peak_errors": peak_errors, "term_shares": shares,
        "score_comparison": [r for r in comparison if r["track"] == "all"],
    }
    write_json(out / "deepspeed_summary.json", summary)
    write_csv(out / "deepspeed_model_states.csv",
              [{k: v for k, v in c.items()} for c in record["calls"]])
    return summary
