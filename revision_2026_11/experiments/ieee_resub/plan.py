#!/usr/bin/env python3
"""Generate the E1-E3 (and smoke) matrices, designs, seed audit and protocol drafts.

`build` is CPU-only and local; it never contacts the cluster. `freeze-final`
must run on the cluster after the approved input copy (PLAN.md, step S0): it
recomputes model/data/container hashes and refuses to write a protocol if any
differs from the earlier frozen records that the draft cites.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import glob
import itertools
import json
from pathlib import Path
import random
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieee_resub import common as C  # noqa: E402

C.bootstrap()
R = C.REPO_ROOT

# ----------------------------------------------------------------------------
# Template sources (cited in every row's `source_configuration`).
REPLICATION_MATRIX = "benchmark/replication_control/replication_control_matrix.csv"
RELEASE_MATRIX = "benchmark/electronics_revision_20261016/release/matrix.csv"
RESERVATION_MATRIX = "benchmark/electronics_revision_20261016/reservation/matrix.csv"
TIMING_MATRIX = "benchmark/electronics_revision_20261016/timing/matrix.csv"
GRANITE_BOUNDARY = "memory_tuner/rlvram_strengthening_granite_confirmatory.csv"
PHI_BOUNDARY = "memory_tuner/rlvram_v2_cross_family_confirmatory.csv"
PAIRED_DURATION = "memory_tuner/rlvram_revision_temporal.csv"

WORKLOAD_FIELDS = (
    "rollout_tp_size", "actor_micro_batch", "rollout_logprob_micro_batch", "ref_logprob_micro_batch",
    "vllm_gpu_memory_utilization", "parameter_offload", "optimizer_offload", "free_cache_engine",
    "max_prompt_length", "max_response_length", "max_model_len", "max_num_seqs", "rollout_n",
    "train_batch_size", "train_max_samples", "val_max_samples", "gpu_monitor_interval_ms",
)

# E1 cases. The original request named Phi/longer-prompt code and Qwen-3B/GSM8K;
# both are predicted to exceed device capacity at micro-batch 16 once every
# response is forced to its cap (see predictions and PLAN.md section 2.3), so the
# same models are kept with MATH, which stays among the twelve replication cases.
E1_CASES = (
    ("ibm-granite/granite-3.3-2b-instruct", "gsm8k"),
    ("microsoft/Phi-4-mini-instruct", "math"),
    ("Qwen/Qwen2.5-3B-Instruct", "math"),
    ("Qwen/Qwen2.5-3B-Instruct", "code_heavy_tail"),
)
E1_CONDITIONS = (("08", "ext"), ("08", "full"), ("16", "ext"), ("16", "full"))

E2_CONFIGS = {
    "q3-gsm8k-rel-r65": (RELEASE_MATRIX, "release-q3-gsm8k-s501-release1-r65", 2,
                         "Qwen2.5-3B/GSM8K release anchor at 0.65 with generation-state release "
                         "(release experiment; release-sensitivity re-execution 2/3 within margin)"),
    "g2-math-c3": (GRANITE_BOUNDARY, "strong-granite-confirmatory-grpo-math_short-c3-s92", 2,
                   "Granite-3.3-2B/MATH c3 (boundary max 38,718 MiB; 194 MiB headroom)"),
    "p4-gsm8k-c3": (PHI_BOUNDARY, "v2cf-confirm-phi4_mini-grpo-gsm8k_long-c3-s42", 2,
                    "Phi-4-mini/GSM8K c3 (boundary 39,550-39,570 MiB; above margin)"),
    "q7-gsm8k-off0-r60": (RESERVATION_MATRIX, "reservation-q7-gsm8k-s452-off0-r60", 4,
                          "Qwen2.5-7B/GSM8K, no actor offload, reservation 0.60, micro-batch 8 "
                          "(reservation panel 3/3 within at 38,560 MiB)"),
}

E3_CONFIGS = {
    # name: (model, dataset, offload, reservation)
    "q7-gsm8k-off1-r70": ("Qwen/Qwen2.5-7B-Instruct", "gsm8k", "True", "0.70"),
    "q7-gsm8k-off0-r60": ("Qwen/Qwen2.5-7B-Instruct", "gsm8k", "False", "0.60"),
    "m7-code_heavy_tail-off1-r70": ("mistralai/Mistral-7B-Instruct-v0.3", "code_heavy_tail", "True", "0.70"),
    "m7-code_heavy_tail-off0-r55": ("mistralai/Mistral-7B-Instruct-v0.3", "code_heavy_tail", "False", "0.55"),
}
E3_TIMEOUTS = {"screen": 2400, "long": 21600, "restore": 12600}
E3_RESTORE_SEED = 631
E3_RESTORE_STEP = 50


def rows_of(relative):
    return C.read_rows(R / relative)


def find(relative, experiment_id):
    rows = [r for r in rows_of(relative) if r["experiment_id"] == experiment_id]
    C.require(len(rows) == 1, f"Template {experiment_id} not found once in {relative}")
    return rows[0]


def blank_row(**values):
    row = {column: "" for column in C.COLUMNS}
    row.update(algorithm="grpo", rollout_tp_size="1", lora_rank="64", lora_alpha="32",
               use_kl_loss="True", gpu_monitor_interval_ms="100", total_epochs="20",
               save_freq="-1", test_freq="-1", val_before_train="False",
               max_actor_ckpt_to_keep="1", resume_mode="disable", resume_step="",
               capture_response_lengths="True", fixed_response_length="False",
               canonical_batch_order="False", shape_trace="False", checkpoint_to_scratch="False",
               custom_reward_function_name="compute_score", study_stage="confirmatory")
    row.update(values)
    return row


def apply_workload(row, template, dataset, model):
    for field in WORKLOAD_FIELDS:
        row[field] = template[field]
    row.update(model=model, model_family=C.FAMILY[model], model_revision=C.REVISIONS[model],
               dataset=dataset, workload_name=dataset,
               data_dir=str(C.CLUSTER_DATA_ROOT / C.DATA_SUBDIRS[dataset]),
               custom_reward_function_path=C.CODE_REWARD if dataset.startswith("code") else "null")
    return row


def instrumentation(row, full):
    row.update(instrumentation="full" if full else "external",
               allocator_trace_enabled=str(bool(full)), allocator_trace_sync=str(int(bool(full))))
    return row


def williams4():
    first = [0, 1, 3, 2]
    return [[(value + shift) % 4 for value in first] for shift in range(4)]


def finalize(rows, experiment):
    for row in rows:
        row["experiment"] = experiment
        row["run_group"] = C.GROUPS[experiment]
        row["pair_id"] = row["block_id"]
        for key in ("experiment_id", "block_id"):
            C.require(C.SAFE_ID.fullmatch(row[key]), f"Unsafe {key}: {row[key]}")
        C.require(set(row) == set(C.COLUMNS), f"Column drift in {row['experiment_id']}")
    C.require(len({r["experiment_id"] for r in rows}) == len(rows), "Duplicate experiment ids")
    return rows


# ----------------------------------------------------------------------------
def e1_rows(seeds=C.SEEDS["e1"], order_seed=C.ORDER_SEEDS["e1"], experiment="e1", cases=E1_CASES,
            conditions=None):
    replication = rows_of(REPLICATION_MATRIX)
    rng = random.Random(order_seed)
    symbols = list(E1_CONDITIONS)
    rng.shuffle(symbols)              # random symbol -> condition mapping
    squares = williams4()
    case_order = list(cases)
    rng.shuffle(case_order)           # which case excludes which Williams row
    plan = []
    for case_index, case in enumerate(case_order):
        sequences = [(case_index + j) % 4 for j in range(len(seeds))]
        rng.shuffle(sequences)
        for seed, sequence in zip(seeds, sequences):
            plan.append((case, seed, sequence))
    rng.shuffle(plan)                 # randomized execution (array) order
    rows = []
    for block_index, ((model, dataset), seed, sequence) in enumerate(plan):
        template = next(r for r in replication if r["model"] == model and r["dataset"] == dataset
                        and r["condition"] == "base" and float(r["vllm_gpu_memory_utilization"]) == 0.6)
        short = C.SHORT[model]
        block = f"{experiment}-{short}-{dataset}-s{seed}"
        ordered = [symbols[k] for k in squares[sequence]]
        if conditions is not None:
            ordered = [c for c in ordered if c in conditions]
        for period, (batch, logging) in enumerate(ordered, 1):
            condition = f"b{batch}-{logging}"
            row = blank_row(
                experiment_id=f"{block}-{condition}", configuration_level="fixed-length-batch-logging",
                case_id=f"{experiment}-{short}-{dataset}", configuration_id=f"{experiment}-{short}-{dataset}-{condition}",
                block_id=block, block_index=str(block_index), period=str(period), condition=condition,
                role="contrast", sequence_id=str(sequence), gpu_count="2", gpu_subset_ordinals="0;1",
                allocation_gpus="2", training_seed=str(seed), total_training_steps="1",
                fixed_response_length="True", canonical_batch_order="True", shape_trace="True",
                per_process_timeout_seconds="1200",
                source_configuration=f"{REPLICATION_MATRIX}:{template['experiment_id']} (reservation 0.60 workload)",
                source_files=REPLICATION_MATRIX,
                novelty="fresh seeds; every response forced to the response cap; identical prompts and order")
            apply_workload(row, template, dataset, model)
            row["actor_micro_batch"] = str(int(batch))
            row["vllm_gpu_memory_utilization"] = "0.60"
            instrumentation(row, logging == "full")
            rows.append(row)
    design = dict(order_seed=order_seed, williams_rows=squares,
                  symbol_to_condition={str(i): f"b{b}-{l}" for i, (b, l) in enumerate(symbols)},
                  case_exclusion_order=[f"{C.SHORT[m]}-{d}" for m, d in case_order],
                  randomization=("Williams 4x4 square (each condition once per position, each directed "
                                 "adjacency once per square); case i receives rows {i, i+1, i+2} mod 4 "
                                 "assigned to its seeds at random, so each row is used three times; "
                                 "block execution order shuffled with the same seed."))
    return finalize(rows, experiment), design


def e2_rows(seeds=C.SEEDS["e2"], order_seed=C.ORDER_SEEDS["e2"]):
    rng = random.Random(order_seed)
    names = list(E2_CONFIGS)
    squares = williams4()
    assignments = [0, 1, 2, 3, 0, 1, 2, 3] + rng.sample(range(4), 2)
    rng.shuffle(assignments)
    blocks = list(zip(seeds, assignments))
    rng.shuffle(blocks)
    rows = []
    for block_index, (seed, sequence) in enumerate(blocks):
        block = f"e2-s{seed}"
        for period, name_index in enumerate(squares[sequence], 1):
            name = names[name_index]
            source, template_id, gpus, description = E2_CONFIGS[name]
            template = find(source, template_id)
            model = template["model"]
            row = blank_row(
                experiment_id=f"e2-{name}-s{seed}", configuration_level="boundary-repeat",
                case_id=f"e2-{name}", configuration_id=f"e2-{name}", block_id=block,
                block_index=str(block_index), period=str(period), condition=name, role="repeat",
                sequence_id=str(sequence), gpu_count=str(gpus),
                gpu_subset_ordinals=";".join(map(str, range(gpus))), allocation_gpus="4",
                training_seed=str(seed), total_training_steps="1", per_process_timeout_seconds="900",
                source_configuration=f"{source}:{template_id} - {description}",
                source_files=source, novelty="ten fresh one-step repeats of a near-boundary configuration")
            apply_workload(row, template, template["dataset"], model)
            # All four originals ran with allocator tracing and per-operation sync
            # (launcher default before 13 Sep 2026; explicit True/1 in the panels).
            instrumentation(row, True)
            for key in ("lora_rank", "lora_alpha", "use_kl_loss"):
                if template.get(key):
                    row[key] = template[key]
            rows.append(row)
    counts = {name: [0] * 4 for name in names}
    for row in rows:
        counts[row["condition"]][int(row["period"]) - 1] += 1
    design = dict(order_seed=order_seed, williams_rows=squares, configuration_symbols=names,
                  row_assignments=dict(blocks), position_counts=counts,
                  randomization=("Williams 4x4 rows over (a,b,c,d); rows 0-3 used twice plus two distinct "
                                 "extra rows drawn with the order seed (10 blocks); block order shuffled."))
    return finalize(rows, "e2"), design


def e3_rows(seeds=C.SEEDS["e3"], order_seed=C.ORDER_SEEDS["e3"]):
    timing = rows_of(TIMING_MATRIX)
    rng = random.Random(order_seed)
    blocks = list(itertools.product(E3_CONFIGS, seeds))
    rng.shuffle(blocks)
    rows = []
    for block_index, (name, seed) in enumerate(blocks):
        model, dataset, offload, reservation = E3_CONFIGS[name]
        template = next(r for r in timing if r["model"] == model and r["dataset"] == dataset
                        and r["condition"] == "off1-r70-release1")
        block = f"e3-{name}-s{seed}"
        roles = ["screen", "long"] + (["restore"] if seed == E3_RESTORE_SEED else [])
        for period, role in enumerate(roles, 1):
            row = blank_row(
                experiment_id=f"{block}-{role}", configuration_level="long-schedule",
                case_id=f"e3-{C.SHORT[model]}-{dataset}", configuration_id=f"e3-{name}",
                block_id=block, block_index=str(block_index), period=str(period), condition=name,
                role=role, sequence_id="0", gpu_count="4", gpu_subset_ordinals="0;1;2;3",
                allocation_gpus="4", training_seed=str(seed),
                total_training_steps="1" if role == "screen" else "100", total_epochs="100",
                save_freq="25", test_freq="20", val_before_train="False" if role == "restore" else "True",
                max_actor_ckpt_to_keep="3", checkpoint_to_scratch="True",
                resume_mode="resume_path" if role == "restore" else "disable",
                resume_source_experiment_id=f"{block}-long" if role == "restore" else "",
                resume_step=str(E3_RESTORE_STEP) if role == "restore" else "",
                per_process_timeout_seconds=str(E3_TIMEOUTS[role]),
                source_configuration=(f"{TIMING_MATRIX}:{template['experiment_id']} (iteration-cost workload; "
                                      f"offload/reservation from the reservation panel); schedule from "
                                      f"{PAIRED_DURATION}"),
                source_files=f"{TIMING_MATRIX};{RESERVATION_MATRIX};{PAIRED_DURATION}",
                novelty={"screen": "fresh one-step screen with the long-run schedule",
                         "long": "fresh 100-step run with validation and checkpoints",
                         "restore": "resume from the step-50 checkpoint of the seed-631 long run"}[role])
            apply_workload(row, template, dataset, model)
            row.update(actor_micro_batch="8", parameter_offload=offload, optimizer_offload="False",
                       vllm_gpu_memory_utilization=reservation, free_cache_engine="True",
                       train_max_samples="-1", val_max_samples="32")
            instrumentation(row, False)
            rows.append(row)
    design = dict(order_seed=order_seed, block_order=[f"e3-{n}-s{s}" for n, s in blocks],
                  within_block="screen (period 1), then long run (period 2), then restore (period 3, "
                               "seed 631 only); fixed order as in the paired duration test",
                  restore=dict(seed=E3_RESTORE_SEED, step=E3_RESTORE_STEP))
    return finalize(rows, "e3"), design


def smoke1_rows():
    """Development only: fixed-length path, shape identity, riskiest E1 case."""
    rows, _ = e1_rows(seeds=C.SEEDS["smoke1"], order_seed=1, experiment="smoke1",
                      cases=(("ibm-granite/granite-3.3-2b-instruct", "gsm8k"),))
    keep = [r for r in rows if r["condition"] in ("b16-full", "b08-ext")]
    risky, _ = e1_rows(seeds=C.SEEDS["smoke1"], order_seed=1, experiment="smoke1",
                       cases=(("Qwen/Qwen2.5-3B-Instruct", "code_heavy_tail"),))
    keep += [r for r in risky if r["condition"] == "b16-full"]
    # Granite b16-full, Granite b08-ext (cross-process shape identity), then the
    # highest predicted actor peak (Qwen-3B/longer-prompt code, b16-full) last.
    order = {("g2", "b16-full"): 1, ("g2", "b08-ext"): 2, ("q3", "b16-full"): 3}
    block = "smoke1-s641"
    for row in keep:
        row.update(block_id=block, block_index="0", period=str(order[(C.SHORT[row["model"]], row["condition"])]),
                   study_stage="development", novelty="development smoke test; excluded from all formal estimates")
    keep.sort(key=lambda r: int(r["period"]))
    return finalize(keep, "smoke1")


def smoke2_rows():
    """Development only: 4-GPU path, validation, checkpoints, retention, restore."""
    rows, _ = e3_rows(seeds=(642,), order_seed=1)
    # e3_rows only creates restore rows for seed 631; build the smoke pair explicitly.
    long_row = next(r for r in rows if r["condition"] == "q7-gsm8k-off1-r70" and r["role"] == "long")
    long_row = copy.deepcopy(long_row)
    block = "smoke2-s642"
    long_row.update(experiment_id=f"{block}-long4", block_id=block, block_index="0", period="1",
                    total_training_steps="4", save_freq="2", test_freq="2", study_stage="development",
                    per_process_timeout_seconds="2700",
                    novelty="development smoke test; excluded from all formal estimates")
    restore = copy.deepcopy(long_row)
    restore.update(experiment_id=f"{block}-restore", period="2", role="restore", resume_mode="resume_path",
                   resume_source_experiment_id=long_row["experiment_id"], resume_step="2",
                   val_before_train="False", per_process_timeout_seconds="2100")
    return finalize([long_row, restore], "smoke2")


# ----------------------------------------------------------------------------
def seed_audit(new_seeds):
    csv.field_size_limit(10 ** 9)
    used = {}
    patterns = ["memory_tuner/*.csv", "benchmark/**/*.csv", "profiles/**/*.csv", "*.csv"]
    for pattern in patterns:
        for name in glob.glob(str(R / pattern), recursive=True):
            path = Path(name)
            if C.EXPERIMENT_DIR in path.parents:
                continue
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                columns = [c for c in (reader.fieldnames or []) if c in ("training_seed", "seed")]
                if not columns:
                    continue
                for row in reader:
                    for column in columns:
                        value = (row.get(column) or "").strip()
                        if re.fullmatch(r"-?\d+", value):
                            used.setdefault(int(value), set()).add(str(path.relative_to(R)))
    collisions = {s: sorted(used[s]) for s in new_seeds if s in used}
    return dict(scanned_patterns=patterns, previously_used=sorted(used), new_seeds=sorted(new_seeds),
                collisions=collisions, fresh=not collisions,
                scope_note="MI250X (LUMI) panels reuse A100 seeds by design; LLMem uses its own seeds "
                           "in benchmark/electronics_revision_20261016/llmem_native/matrix.csv (scanned).")


def expected_inputs():
    """Hashes frozen by earlier protocols; the final freeze must reproduce them."""
    models, data = {}, {}
    for name in glob.glob(str(R / "benchmark/**/*.json"), recursive=True):
        try:
            record = json.loads(Path(name).read_text())
        except (OSError, ValueError):
            continue

        def walk(value):
            if isinstance(value, dict):
                if isinstance(value.get("files_sha256"), dict) and value.get("path"):
                    match = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})", value["path"])
                    if match:
                        model = f"{match[1]}/{match[2]}"
                        if C.REVISIONS.get(model) == match[3]:
                            prior = models.setdefault(model, dict(revision=match[3], files_sha256=value["files_sha256"],
                                                                  sources=[]))
                            C.require(prior["files_sha256"] == value["files_sha256"], f"Inconsistent {model} hashes")
                            prior["sources"].append(str(Path(name).relative_to(R)))
                for key, item in value.items():
                    if isinstance(key, str) and key.endswith(".parquet"):
                        digest = item if isinstance(item, str) else (item or {}).get("sha256") if isinstance(item, dict) else None
                        relative = key.replace(str(C.LEGACY_RL_ROOT) + "/", "")
                        if isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest) and relative.startswith("data/"):
                            prior = data.setdefault(relative, dict(sha256=digest, sources=[]))
                            C.require(prior["sha256"] == digest, f"Inconsistent {relative} hash")
                            prior["sources"].append(str(Path(name).relative_to(R)))
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(record)
    for entry in list(models.values()) + list(data.values()):
        entry["sources"] = sorted(set(entry["sources"]))
    return dict(models=models, data=data)


def source_hashes():
    files = [C.LAUNCHER, C.EXPERIMENT_RELATIVE / "verl_patch/ieee_resub_verl.patch",
             Path("run_grpo_instrumented.slurm"), Path("memory_tuner/run_matched_gpu.py"),
             Path("memory_tuner/device_contract.py"), Path("memory_tuner/durable_metrics.py"),
             Path("memory_tuner/gpu_monitor.py"), Path("memory_tuner/capture_environment.py"),
             Path("memory_tuner/code_shape_reward.py"), Path("memory_tuner/log_parser.py"),
             Path("memory_tuner/electronics_failure_evidence.py"), Path("memory_tuner/select_port_block.py"),
             Path("memory_tuner/electronics_experiment.py"), Path("memory_tuner/run_offload_factorial.py"),
             Path("memory_tuner/host_memory_monitor.py"), Path("memory_tuner/resolve_hf_snapshot.py"),
             Path("memory_tuner/grpo_raw_evidence.py"), Path("memory_tuner/benchmark_v2_env.py")]
    files += sorted(p.relative_to(R) for p in (C.EXPERIMENT_DIR / "ieee_resub").glob("*.py"))
    files += sorted(p.relative_to(R) for p in (C.EXPERIMENT_DIR / "slurm").glob("*.slurm"))
    hashes = {str(p): C.digest(R / p) for p in files}
    hashes[str(C.INSTRUMENTED_PACKAGES) + "/ (tree)"] = C.tree_digest(R / C.INSTRUMENTED_PACKAGES)
    return hashes


def protocol(study, rows, design, *, mode, inputs, predictions=None):
    caps = dict(C.CAPS[study])
    models = sorted({r["model"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    data_expected = {}
    for dataset in datasets:
        for split in ("train.parquet", "test.parquet"):
            relative = f"data/{C.DATA_SUBDIRS[dataset]}/{split}"
            data_expected[relative] = inputs["data"][relative]["sha256"]
    return dict(
        schema_version=1, study=study, status=mode, group=C.GROUPS[study],
        development=study.startswith("smoke"),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        account=C.ACCOUNT, partition=C.PARTITION, caps=caps,
        planned_invocations=len(rows), blocks=len({r["block_id"] for r in rows}),
        allocation_gpus=caps["allocation_gpus"], capacity_mib=C.CAPACITY_MIB, margin_mib=C.MARGIN_MIB,
        container=dict(name=C.CONTAINER_NAME, sha256=C.CONTAINER_SHA256, torch=C.TORCH_VERSION, vllm=C.VLLM_VERSION),
        model_cache={m: dict(revision=C.REVISIONS[m],
                              path=str(C.CLUSTER_HF_HUB / ("models--" + m.replace("/", "--")) / "snapshots" / C.REVISIONS[m]),
                              files_sha256=inputs["models"][m]["files_sha256"],
                              expected_from=inputs["models"][m]["sources"]) for m in models},
        data_root=str(C.CLUSTER_DATA_ROOT), data_sha256=data_expected,
        design=design, predictions=predictions,
        measurement="Per-task-device NVML every 100 ms; host stage markers are not allocation ownership.",
        failure_policy=("Only diagnosed GPU-memory errors (OOM or startup rejection) are memory failures; "
                        "timeouts, host OOM and missing evidence stay unresolved."),
        success_policy=("Successful trainer exit, all requested durable step records, physical-device identity, "
                        "valid raw traces, parameter inventory, and every scheduled validation/checkpoint."),
        replacement_policy="No automatic retries or replacements; any rerun needs a written amendment first.",
        no_automatic_retries=True,
        source_files_sha256=source_hashes(),
    )


E1_PREDICTION_BASIS = (
    "Per-model linear fit of the actor-stage peak on micro-batch tokens (b x realized mean prompt+response "
    "length) over the 144 original factorial runs (profiles/standard_grpo/factorial-trials.csv joined with "
    "profiles/review_revision/realized-workload-trials.csv); forced length = realized mean prompt + response cap. "
    "Descriptive expectations only; never an outcome requirement.")
E1_FIT = {"granite33_2b": (10612, 0.394, 268), "phi4_mini": (14860, 1.156, 673), "qwen25_3b": (12068, 0.794, 496)}
E1_PROMPT_MEAN = {("granite33_2b", "gsm8k"): 146, ("phi4_mini", "math"): 76, ("qwen25_3b", "math"): 105,
                  ("qwen25_3b", "code_heavy_tail"): 720, ("phi4_mini", "code_heavy_tail"): 664,
                  ("qwen25_3b", "gsm8k"): 107}
E1_RESPONSE_CAP = {"gsm8k": 2048, "math": 512, "code_heavy_tail": 1024}
E1_NATURAL_WHOLE_RUN_R60 = {("granite33_2b", "gsm8k"): 32218, ("phi4_mini", "math"): 36128,
                            ("qwen25_3b", "math"): 33458, ("qwen25_3b", "code_heavy_tail"): 33316,
                            ("phi4_mini", "code_heavy_tail"): 35866, ("qwen25_3b", "gsm8k"): 33036}


def e1_predictions():
    out = []
    for family, dataset in list(E1_PROMPT_MEAN):
        a, c, sd = E1_FIT[family]
        length = E1_PROMPT_MEAN[(family, dataset)] + E1_RESPONSE_CAP[dataset]
        b8, b16 = a + c * 8 * length, a + c * 16 * length
        whole8 = max(b8, E1_NATURAL_WHOLE_RUN_R60[(family, dataset)])
        whole16 = max(b16, E1_NATURAL_WHOLE_RUN_R60[(family, dataset)])
        selected = (family, dataset) in {(C.FAMILY[m], d) for m, d in E1_CASES}
        out.append(dict(model_family=family, dataset=dataset, selected=selected,
                        forced_sequence_tokens=length, actor_peak_b8_mib=round(b8), actor_peak_b16_mib=round(b16),
                        actor_contrast_mib=round(b16 - b8), whole_run_contrast_mib=round(whole16 - whole8),
                        residual_sd_mib=sd, exceeds_margin_b16=b16 > C.MARGIN_MIB,
                        exceeds_capacity_b16=b16 > C.CAPACITY_MIB))
    return dict(basis=E1_PREDICTION_BASIS, cases=out)


def build(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = expected_inputs()
    for model in C.REVISIONS:
        C.require(model in inputs["models"], f"No frozen file hashes found for {model}")
    built = {}
    rows, design = e1_rows(); built["e1"] = (rows, design, e1_predictions())
    rows, design = e2_rows(); built["e2"] = (rows, design, None)
    rows, design = e3_rows(); built["e3"] = (rows, design, None)
    built["smoke1"] = (smoke1_rows(), dict(note="development"), e1_predictions())
    built["smoke2"] = (smoke2_rows(), dict(note="development"), None)
    for study, (rows, _, _) in built.items():
        C.write_rows(out_dir / f"{study}_matrix.csv", rows, C.COLUMNS)
    # The outer Slurm scripts depend on block counts and are hashed into the drafts.
    from ieee_resub.render import write_slurm
    write_slurm(out_dir)
    summary = {}
    all_rows = []
    for study, (rows, design, predictions) in built.items():
        draft = protocol(study, rows, design, mode="draft-awaiting-approval", inputs=inputs, predictions=predictions)
        draft["matrix_sha256"] = C.digest(out_dir / f"{study}_matrix.csv")
        C.write_json(out_dir / "protocol_draft" / f"{study}.json", draft)
        summary[study] = dict(invocations=len(rows), blocks=draft["blocks"],
                              matrix_sha256=draft["matrix_sha256"],
                              protocol_draft_sha256=C.digest(out_dir / "protocol_draft" / f"{study}.json"))
        all_rows += rows
    C.write_rows(out_dir / "all_planned_slots.csv", all_rows, C.COLUMNS)
    formal = [int(r["training_seed"]) for s in C.EXPERIMENTS for r in built[s][0]]
    smoke = [int(r["training_seed"]) for s in ("smoke1", "smoke2") for r in built[s][0]]
    audit = seed_audit(sorted(set(formal) | set(smoke)))
    C.write_json(out_dir / "seed_audit.json", audit)
    C.require(audit["fresh"], f"Seed collision: {audit['collisions']}")
    C.write_json(out_dir / "expected_inputs.json", inputs)
    C.write_json(out_dir / "build_summary.json", summary)
    return summary


def freeze_final(study: str, out_dir: Path, matrices: Path):
    """Cluster-only: recompute input hashes and write the immutable protocol."""
    rows = C.read_rows(matrices / f"{study}_matrix.csv")
    draft = json.loads((matrices / "protocol_draft" / f"{study}.json").read_text())
    C.require(C.digest(matrices / f"{study}_matrix.csv") == draft["matrix_sha256"], "Matrix differs from draft")
    for model, entry in draft["model_cache"].items():
        snapshot = Path(entry["path"])
        for name, expected in entry["files_sha256"].items():
            C.require(C.digest(snapshot / name) == expected, f"{model}:{name} differs from frozen record")
    for relative, expected in draft["data_sha256"].items():
        C.require(C.digest(C.CLUSTER_PROJECT_ROOT / relative) == expected, f"{relative} differs")
    image = C.CLUSTER_PROJECT_ROOT / "images" / C.CONTAINER_NAME
    C.require(C.digest(image) == C.CONTAINER_SHA256, "Container differs")
    C.require(draft["source_files_sha256"] == source_hashes(), "Source files differ from the approved draft")
    C.require(str(out_dir).startswith(str(C.CLUSTER_PROJECT_ROOT / "protocols")),
              "Final protocols live outside the immutable source snapshot")
    final = dict(draft, status="frozen", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                 draft_sha256=C.digest(matrices / "protocol_draft" / f"{study}.json"),
                 analysis_plan_sha256=C.digest(C.EXPERIMENT_DIR / "PLAN.md"))
    target = out_dir / f"{study}_protocol.json"
    C.write_json(target, final, exclusive=True)
    target.chmod(0o444)
    print(json.dumps(dict(study=study, protocol=str(target), protocol_sha256=C.digest(target))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=C.EXPERIMENT_DIR / "matrices")
    f = sub.add_parser("freeze-final")
    f.add_argument("--study", choices=C.ALL_STUDIES, required=True)
    f.add_argument("--matrices", type=Path, default=C.EXPERIMENT_DIR / "matrices")
    f.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.out), indent=2))
    else:
        freeze_final(args.study, args.out, args.matrices)


if __name__ == "__main__":
    main()
