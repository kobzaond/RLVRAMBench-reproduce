"""Prospective completion replication: preview, then explicit parent-approved freeze.

No downloads, fitting, submissions, retries, or edits to historical inputs.
The four-file freeze is created only by ``freeze`` or the single anchored
``refreeze-implementation`` command. Historical freeze bytes are never replaced.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import itertools
import json
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import time

import numpy as np

from memory_tuner import estimation_baselines as eb
from memory_tuner.device_contract import digest, sync_directory, write_new_json
from memory_tuner.model_memory_metadata import extract
from memory_tuner.resolve_hf_snapshot import validate_snapshot

GROUP = "completion_replication"
VERSION = "completion-replication-1"
DIRECTORY = "benchmark/completion_replication"
CANONICAL_ROOT = Path("/mnt/proj3/open-35-44/verl/rl")
MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "slug": "qwen25-7b", "family": "qwen25",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "slug": "mistral-7b", "family": "mistral",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    },
}
# Local adapter map; never assign to eb.MODEL_FAMILIES or change the old module.
FAMILIES = {**eb.MODEL_FAMILIES, **{m: s["family"] for m, s in MODELS.items()}}
DATASETS = ("gsm8k", "math", "code_heavy_tail")
DATA_ROOTS = dict(gsm8k="data/gsm8k", math="data/math",
                  code_heavy_tail="data/codecontests/heavy_tail")
LEVELS = {"c1": (4, .45, "True"), "c2": (8, .6, "False"), "c3": (16, .7, "False")}
LEVEL_GROUPS = {"c1_offload": ("c1",), "c2_c3_resident_actor": ("c2", "c3")}
SEEDS = (221, 222, 223)
ORDER_SEEDS = (2026091621, 2026091622, 2026091623)
CAPS = dict(initial_allocations=54, max_concurrent_allocations=3, gpus_per_allocation=4,
            cpus_per_task=64, host_memory_mib=393216, wall_minutes_per_allocation=55,
            payload_timeout_seconds=1500, idle_timeout_seconds=180,
            maximum_reserved_gpu_hours=198, retries=0, requeue=False, replacement=False)
ORIGINAL_PROTOCOL_SHA256 = "1694e447e842600612604891e7b3d036ff0791fa84c4bdbeba54ed093feefdeb"
ORIGINAL_SEAL_SHA256 = "5ed714caff922a5109346a4242545d19f2824636d0f52bb0bbd2a09b40ed4666"
FREEZE_FILES = ("protocol.json", "matrix.csv", "predictions.json")
INITIAL_DIRECTORY = "benchmark/completion_replication_initial"
INITIAL_FREEZE_SHA256 = "632ecc6c4fa410e4e3bcda0847594666144213b8f7292e9bfc106096b9360c60"
INITIAL_FILES_SHA256 = {
    "protocol.json": "252ad9ee9ff28d7752cca521b49c663bf94bb241fbcdd7a1794da7eeda3d1159",
    "matrix.csv": "7ced80768bdf13f663a241ee18c047620d7448d3230b6f51e1a26a5298c2b84a",
    "predictions.json": "c48a51707f24a9e713ab09b1295112cdd935e52445fce740d129b1308a76977e",
}
RETENTION_IMPLEMENTATION_FILES = {
    "run_grpo_instrumented.slurm", "memory_tuner/completion_replication_protocol.py",
    "memory_tuner/run_completion_smoke.py",
}
# A single post-start implementation amendment, not a new scientific freeze.
RUNTIME_AMENDMENT_VERSION = "completion-retention-noclobber-1"
EXECUTION_FREEZE_SHA256 = "ba3e24e9e3a13b60ca35b22f92ec60d5dac322a80d8f68e660bfd3e98ef1f922"
EXECUTION_SOURCE_COMMIT = "8c1ebe4ae084b24c77ae4eb316270b794cdf0c36"
EXECUTION_SOURCE_ROOT = "/mnt/proj3/open-35-44/verl/experiments/completion-replication-8c1ebe4"
FIRST_JOB_ID = "5060748"
AMENDED_INDICES = tuple(range(1, 54))
AMENDMENT_RUNTIME_FILES = frozenset((
    "run_grpo_instrumented.slurm", "memory_tuner/completion_replication_protocol.py",
    "memory_tuner/run_completion_replication.py", "memory_tuner/collect_completion_replication.py",
))
AMENDMENT_SCOPE = (
    "Post-start, outcome-motivated retention implementation amendment only: allow find to "
    "write the exclusively created mktemp inventory under noclobber. Pair index 0 retains "
    "its original source and all original outcomes, without retry or replacement. Only "
    "still-unclaimed indices 1-53 may use the amended source. The initial and execution "
    "freezes, matrix, forecasts, original 90-fit, settings, labels and thresholds remain "
    "unchanged. Neither diagnostics nor evaluation outcomes enter fitting or settings. "
    "The original before-evaluation source seal does not cover this amendment."
)
TRANSFER_QUEUE_DIRECTORY = "third_party/TransferQueue"
TRANSFER_QUEUE_REVISION = "5cb184efd99addd4b40ff25a700aa260ed2e9eed"
RUNTIME_ROOTS = ("memory_tuner", "instrumented_python_packages", TRANSFER_QUEUE_DIRECTORY)
ESSENTIAL_RUNTIME_FILES = frozenset((
    "benchmark.py", "run_grpo_instrumented.slurm",
    "memory_tuner/completion_replication_protocol.py",
    "memory_tuner/run_completion_replication.py",
    "memory_tuner/collect_completion_replication.py",
    "memory_tuner/run_completion_replication.slurm",
    "memory_tuner/estimation_baselines.py", "memory_tuner/model_memory_metadata.py",
    "memory_tuner/run_matched_gpu.py", "memory_tuner/collect_estimation_study.py",
    "memory_tuner/benchmark_v2_env.py", "memory_tuner/device_contract.py",
    "memory_tuner/durable_metrics.py", "memory_tuner/capture_environment.py",
    "memory_tuner/resolve_hf_snapshot.py", "memory_tuner/select_port_block.py",
    "memory_tuner/gpu_monitor.py", "memory_tuner/code_shape_reward.py",
    "memory_tuner/grpo_raw_evidence.py", "memory_tuner/log_parser.py",
    "memory_tuner/scientific_revision_analysis.py",
    "instrumented_python_packages/verl/__init__.py",
    "instrumented_python_packages/verl/trainer/main_ppo.py",
    "instrumented_python_packages/verl/trainer/constants_ppo.py",
    "instrumented_python_packages/verl/trainer/ppo/ray_trainer.py",
    "instrumented_python_packages/verl/utils/logger/aggregate_logger.py",
    "instrumented_python_packages/verl/workers/engine_workers.py",
    "sppo_replay/allocator_trace.py",
    "third_party/TransferQueue/transfer_queue/__init__.py",
    "third_party/TransferQueue/transfer_queue/client.py",
    "third_party/TransferQueue/transfer_queue/controller.py",
    "third_party/TransferQueue/transfer_queue/interface.py",
    "third_party/TransferQueue/transfer_queue/metadata.py",
))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def original_inputs(root):
    """Externally anchored old predictor and numeric/source inputs, read verbatim."""
    directory = root / "benchmark/estimation"
    require(digest(directory / "protocol.json") == ORIGINAL_PROTOCOL_SHA256
            and digest(directory / "prediction_freeze.json") == ORIGINAL_SEAL_SHA256,
            "Original protocol/prediction seal changed")
    seal = json.loads((directory / "prediction_freeze.json").read_text())
    hashes = {**seal["effective_input_sha256"],
              "benchmark/estimation/protocol.json": ORIGINAL_PROTOCOL_SHA256,
              "benchmark/estimation/prediction_freeze.json": ORIGINAL_SEAL_SHA256,
              **{"benchmark/estimation/" + n: h for n, h in seal["files_sha256"].items()},
              **{"benchmark/estimation/" + n: h for n, h in seal["amendments_sha256"].items()}}
    for name, expected in hashes.items():
        require(digest(root / name) == expected, f"Changed inherited input: {name}")
    require(digest(Path(eb.__file__)) == seal["implementation_sha256"],
            "Executed old predictor differs from frozen implementation")
    fitted = json.loads((directory / "fitted_model.json").read_text())
    source = eb.source_records(root, set(fitted["training_configuration_ids"]))
    require(len(source) == 90, "The inherited pool must have exactly 90 configurations")
    metadata = json.loads((directory / "model_metadata.json").read_text())
    return source, fitted, metadata, hashes


def donor(target, source, metadata):
    spec = target["settings"]
    pool = [r for r in source if r["settings"]["workload_id"] == spec["workload_id"]
            and r["settings"]["configuration_level"] == spec["configuration_level"]]
    require(bool(pool), "No donor with matching original workload and compound level")
    require(spec["model_id"] in FAMILIES, "Undeclared target model family")

    def distance(row):
        candidate = row["settings"]
        return (
            candidate["gpu_count"] != spec["gpu_count"],
            FAMILIES[candidate["model_id"]] != FAMILIES[spec["model_id"]],
            abs(np.log(metadata[candidate["model_id"]]["floating_checkpoint_elements"]
                       / metadata[spec["model_id"]]["floating_checkpoint_elements"])),
            candidate["configuration_id"],
        )
    return min(pool, key=distance)


def predict(targets, source, metadata, fitted):
    """Equivalent frozen numeric predictor, with an explicit Mistral donor prior."""
    require(fitted["feature_names"] == list(eb.FEATURES), "Changed feature order")
    require({r["settings"]["configuration_id"] for r in source}
            == set(fitted["training_configuration_ids"]), "Changed donor/fitting pool")
    require(not {t["settings"]["configuration_id"] for t in targets}
            & set(fitted["training_configuration_ids"]), "Target leaked into fitted pool")
    x = eb.features(targets, metadata)
    regression = np.column_stack([np.ones(len(targets)),
                                  x / np.asarray(fitted["component_divisors"])])
    peaks = regression @ np.asarray(fitted["component_coefficients_gib"]) * 1024
    scaled = (x - np.asarray(fitted["scaler_mean"])) / np.asarray(fitted["scaler_scale"])
    logits = scaled @ np.asarray(fitted["classifier_coefficients"]).T + fitted["classifier_intercepts"]
    classifications = [fitted["classifier_classes"][i] for i in np.argmax(logits, axis=1)]
    rows = []
    for index, target in enumerate(targets):
        spec, selected = target["settings"], donor(target, source, metadata)
        for method in eb.METHODS:
            peak = (float(peaks[index]) if method == "component_regression" else
                    selected["completed_peak_mib"] if method == "donor_copy" else None)
            diagnostic = eb.startup_diagnostic(spec, metadata)
            raw_state = eb.peak_state(peak, spec) if method == "component_regression" else None
            state = (("memory_failure" if diagnostic["predicted_startup_budget_failure"]
                      else raw_state) if method == "component_regression" else
                     selected["state"] if method == "donor_copy" else classifications[index])
            rows.append(dict(
                configuration_id=spec["configuration_id"], method=method,
                predicted_state=state, predicted_peak_mib=peak,
                peak_only_predicted_state=raw_state,
                startup_guard_changed_state=state != raw_state if raw_state is not None else None,
                donor_configuration_id=selected["settings"]["configuration_id"]
                if method == "donor_copy" else None,
                donor_gpu_count_matches=selected["settings"]["gpu_count"] == spec["gpu_count"]
                if method == "donor_copy" else None,
                startup_budget_diagnostic=diagnostic))
    return rows


def verify_old_predictions(root, source, metadata, fitted):
    targets = [{"settings": r} for r in eb.read_csv(root / "benchmark/estimation/targets.csv")]
    observed = predict(targets, source, metadata, fitted)
    old = eb.predict(targets, source, metadata, fitted)
    stored = json.loads((root / "benchmark/estimation/predictions.json").read_text())
    require(observed == old == stored, "Adapter does not exactly reproduce old Qwen predictions")


def planned(root, data_root=CANONICAL_ROOT):
    source, _, _, _ = original_inputs(root)
    old_rows = eb.read_csv(root / "benchmark/estimation/matrix.csv")
    subsets = list(itertools.combinations(range(4), 2)) * 9
    random.Random(ORDER_SEEDS[1]).shuffle(subsets)
    order = random.Random(ORDER_SEEDS[0])
    rows, targets = [], []
    for cell, (model, dataset, level) in enumerate(itertools.product(MODELS, DATASETS, LEVELS)):
        model_spec = MODELS[model]
        template = dict(next(r["settings"] for r in source
                             if r["settings"]["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
                             and r["settings"]["legacy_dataset_id"] == dataset
                             and r["settings"]["configuration_level"] == level
                             and r["settings"]["gpu_count"] == "2"))
        micro, reservation, offload = LEVELS[level]
        require((int(template["actor_micro_batch"]),
                 float(template["vllm_gpu_memory_utilization"]), template["parameter_offload"])
                == (micro, reservation, offload), "Original compound-level settings differ")
        common = dict(actor_micro_batch=str(micro), vllm_gpu_memory_utilization=str(reservation),
                      parameter_offload=offload, optimizer_offload="False", free_cache_engine="True",
                      allocator_trace_enabled="True", allocator_trace_sync="1",
                      save_freq="-1", test_freq="-1", val_before_train="False",
                      resume_mode="disable", max_actor_ckpt_to_keep="1", total_training_steps="1")
        for count in (2, 4):
            target = dict(template, **common)
            target.update(configuration_id=f"cr-{model_spec['slug']}-{dataset}-{level}-{count}gpu",
                          case_id=f"cr-{model_spec['slug']}-{dataset}-{count}gpu",
                          study=GROUP, model_id=model, model_family=model_spec["family"],
                          model_name=model.split("/")[-1], gpu_count=str(count))
            targets.append(target)
        firsts = [2, 4, 2] if cell % 2 == 0 else [4, 2, 4]
        order.shuffle(firsts)
        for offset, (seed, first) in enumerate(zip(SEEDS, firsts)):
            pair = f"cr-{model_spec['slug']}-{dataset}-{level}-s{seed}"
            for period, count in enumerate((first, 6 - first), 1):
                base = dict(next(r for r in old_rows
                                 if r["dataset"] == dataset and r["gpu_count"] == str(count)))
                base.update(common)
                base.update(
                    experiment_id=f"{pair}-{count}gpu", pair_id=pair, period=str(period),
                    condition=f"{count}gpu", configuration_level=level,
                    configuration_id=f"cr-{model_spec['slug']}-{dataset}-{level}-{count}gpu",
                    case_id=f"cr-{model_spec['slug']}-{dataset}-{count}gpu",
                    run_group=GROUP, study_stage=GROUP, revision_study=GROUP,
                    model=model, model_family=model_spec["family"], model_revision=model_spec["revision"],
                    training_seed=str(seed), data_dir=str(data_root / DATA_ROOTS[dataset]),
                    gpu_subset_ordinals=";".join(map(str, subsets[cell * 3 + offset]
                                                   if count == 2 else range(4))),
                    custom_reward_function_path=str(data_root / "memory_tuner/code_shape_reward.py")
                    if dataset == "code_heavy_tail" else "null",
                    custom_reward_function_name="compute_score",
                    capture_response_lengths="False")
                rows.append(base)
    pairs = list(dict.fromkeys(r["pair_id"] for r in rows))
    random.Random(ORDER_SEEDS[2]).shuffle(pairs)
    rows.sort(key=lambda r: (pairs.index(r["pair_id"]), int(r["period"])))
    require(len(rows) == 108 and len(targets) == 36 and len(pairs) == 54, "Incomplete plan")
    return rows, targets


def contained_runtime_path(root, name, *, directory=False):
    """Reject noncanonical names and links before opening any sealed source bytes."""
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name),
            f"Unsafe runtime relative name: {name!r}")
    relative = Path(name)
    require(not relative.is_absolute() and relative.as_posix() == name
            and not {".", ".."} & set(relative.parts), f"Unsafe runtime relative name: {name!r}")
    root = Path(root).resolve()
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), f"Linked runtime path: {name}")
    require(path.resolve().is_relative_to(root), f"Runtime path escapes source root: {name}")
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"Missing runtime file/directory: {name}") from error
    require(stat.S_ISDIR(info.st_mode) if directory else
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            f"Runtime path must be an unlinked {'directory' if directory else 'regular file'}: {name}")
    return path


def runtime_python_names(root, directory):
    """Freeze-time discovery only; do not silently omit a linked Python subtree."""
    top = contained_runtime_path(root, directory, directory=True)
    names = set()
    for current, directories, files in os.walk(top, followlinks=False):
        for name in directories:
            require(not (Path(current) / name).is_symlink(), "Linked runtime Python directory")
        for name in files:
            if name.endswith(".py") and not name.startswith("test_"):
                relative = (Path(current) / name).relative_to(Path(root).resolve()).as_posix()
                contained_runtime_path(root, relative)
                names.add(relative)
    return names


def validate_transfer_queue(root):
    """Freeze only: require every pinned Python blob in the populated local submodule."""
    entry = TRANSFER_QUEUE_DIRECTORY + "/transfer_queue/__init__.py"
    contained_runtime_path(root, entry)
    dependency = contained_runtime_path(root, TRANSFER_QUEUE_DIRECTORY, directory=True)

    def git(directory, *arguments):
        return subprocess.check_output(
            ["git", "-C", str(directory), *arguments], text=True, timeout=30)

    gitlink = git(root, "ls-tree", "HEAD", "--", TRANSFER_QUEUE_DIRECTORY).strip()
    require(gitlink == f"160000 commit {TRANSFER_QUEUE_REVISION}\t{TRANSFER_QUEUE_DIRECTORY}",
            "TransferQueue gitlink differs from the pinned revision")
    identity = git(dependency, "rev-parse", "--show-toplevel", "HEAD").splitlines()
    require(identity == [str(dependency), TRANSFER_QUEUE_REVISION],
            "TransferQueue must be a populated checkout at the pinned revision")
    expected = {}
    for record in git(dependency, "ls-tree", "-rz", "--full-tree", "HEAD").split("\0"):
        if not record:
            continue
        info, name = record.split("\t", 1)
        if not name.endswith(".py") or Path(name).name.startswith("test_"):
            continue
        mode, kind, blob = info.split()
        require(mode in ("100644", "100755") and kind == "blob",
                f"Linked or invalid pinned TransferQueue entry: {name}")
        contained_runtime_path(root, TRANSFER_QUEUE_DIRECTORY + "/" + name)
        expected[name] = blob
    actual = runtime_python_names(root, TRANSFER_QUEUE_DIRECTORY)
    require(bool(expected) and actual == {TRANSFER_QUEUE_DIRECTORY + "/" + n for n in expected},
            "Missing or extra Python files in the pinned TransferQueue closure")
    names = sorted(expected)
    observed = git(dependency, "hash-object", "--no-filters", "--", *names).splitlines()
    require(observed == [expected[name] for name in names],
            "TransferQueue Python bytes differ from the pinned revision")


def runtime_hashes(root):
    """Capture the full existing closure once; later additions are outside this seal."""
    validate_transfer_queue(root)
    names = set(ESSENTIAL_RUNTIME_FILES)
    for directory in RUNTIME_ROOTS:
        names.update(runtime_python_names(root, directory))
    return {name: digest(contained_runtime_path(root, name)) for name in sorted(names)}


def verify_runtime(root, hashes):
    """Verify exact sealed names, without Git, live dependencies, or a fresh inventory."""
    require(isinstance(hashes, dict) and ESSENTIAL_RUNTIME_FILES.issubset(hashes),
            "Missing essential completion/hook runtime names in the source seal")
    for name, expected in hashes.items():
        require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected),
                f"Invalid sealed runtime SHA256: {name}")
        path = contained_runtime_path(root, name)
        require(digest(path) == expected,
                f"Changed sealed runtime source: {name}; replay requires the archived source snapshot")


def cached_metadata(hf_home, inherited, *, hash_payloads=True):
    metadata, cache = dict(inherited), {}
    for model, spec in MODELS.items():
        path = validate_snapshot(hf_home / "hub" / ("models--" + model.replace("/", "--"))
                                 / "snapshots" / spec["revision"])
        require(path.name == spec["revision"], "Cache resolved to another model revision")
        measured = extract(path, model)
        if model in inherited:
            require(measured == inherited[model], "Cached Qwen metadata changed")
        metadata[model] = measured
        # Header hashes bind numeric features. Full payload SHA256 binds actual weights;
        # hash only at freeze and once per pair, not every collection or prediction.
        files = ({p.name: digest(p) for p in sorted(path.iterdir()) if p.is_file()}
                 if hash_payloads else {})
        cache[model] = dict(path=str(path), revision=spec["revision"], files_sha256=files)
    return metadata, cache


def validate_cache(protocol, model):
    record = protocol["model_cache"][model]
    path = validate_snapshot(Path(record["path"]))
    require(path.name == MODELS[model]["revision"], "Model cache revision changed")
    require({p.name for p in path.iterdir() if p.is_file()} == set(record["files_sha256"]),
            "Model cache inventory changed")
    for name, expected in record["files_sha256"].items():
        require(digest(path / name) == expected, f"Changed cached model file: {model}/{name}")
    return path


def specification():
    return dict(
        schema_version=VERSION, group=GROUP, models=MODELS, datasets=list(DATASETS),
        compound_levels={k: list(v) for k, v in LEVELS.items()},
        evaluation_seeds=list(SEEDS), order_seeds=list(ORDER_SEEDS),
        planned_pairs=54, planned_processes=108, target_configurations=36,
        resource_cap=CAPS, allocation_gpus=4, task_gpu_counts=[2, 4],
        device_capacity_mib=40960, memory_margin_limit_mib=38912,
        original_protocol_sha256=ORIGINAL_PROTOCOL_SHA256,
        original_prediction_seal_sha256=ORIGINAL_SEAL_SHA256,
        transfer_queue_revision=TRANSFER_QUEUE_REVISION,
        source_closure_policy="Freeze the full existing non-test Python execution/collector closure, "
        "all essential completion/logger/device/allocator hooks, and populated pinned TransferQueue "
        "Python bytes. Verify exactly those sealed relative names thereafter. Unrelated later "
        "additions are allowed; changes to any sealed file require replay from an archived source "
        "snapshot containing the original bytes. Archive that snapshot with the protocol and raw evidence.",
        interpretation="Outcome-motivated new measurement implementation after visibility of the "
        "two old panels. Both historical panels retain their original per-configuration labels "
        "and unresolved statuses and remain separate from this cohort. This is "
        "prospective fresh-seed replication; no retrospective rescue or unseen model selection.",
        diagnostic_visibility="Parent separately runs a Qwen7/GSM8K/c2/four-GPU diagnostic with "
        "seed191 to check durable capture. It is excluded from fitting, selection, target slots "
        "and costs here. CPU prompt-format and filtering checks are visible before freezing; "
        "their outputs do not enter predictors. No Mistral training or GPU-memory outcomes "
        "precede this new forecast freeze.",
        predictions="Reuse original 90-configuration fitted_model.json verbatim; no refit, retune, "
        "target outcomes, measured lengths, or new-panel outcomes enter prediction.",
        donor_rule="Same original workload and compound level; prefer same GPU count, then same "
        "family, then absolute log parameter-count distance, then configuration_id. Mistral is "
        "explicitly a new family: all historical candidates mismatch its family, so fall back "
        "to GPU count, parameter distance, and ID. Qwen predictions must reproduce exactly.",
        analysis_breakdown="Report model-level scores and separate c1 (actor parameter offload "
        "enabled) from c2/c3 (disabled), including unresolved approvals. Do not attribute c1 "
        "recovery to successful prediction of resident-actor settings. Mark the ten targets "
        "with mismatched donor GPU counts: six four-GPU c1 targets and four four-GPU code "
        "targets at c2/c3. These transfer GPU count as well as model.",
        level_groups={k: list(v) for k, v in LEVEL_GROUPS.items()},
        completion="Zero wrapper/trial exit, nonempty immutable raw console, exact durable metric "
        "from real LocalLogger after normal print, complete physical-device worker coverage, "
        "source/model/data/stack provenance, complete traces and trainer lifecycle. Never "
        "synthesize a metric or use an old exit-only/log-free completion rule.",
        outcome_rule="Three eligible distinct new seeds required per configuration. Any validated "
        "memory failure implies memory_failure; else max completed peak >38912 MiB implies "
        "above_margin; else within_margin. Missing/invalid/timeout slots remain unresolved. "
        "Partial known failure/margin flags remain visible; no peak imputation for failure.",
        failure_policy="One permanent pair claim in the canonical raw root. No retry/requeue/"
        "replacement. Continue period two after any outcome only with successful cleanup, "
        "all-four-device idle gates, and a full 1500-second slot plus cleanup reserve remaining.",
        randomization="27 two-first and 27 four-first; 2:1 within each three-seed cell. Each "
        "two-of-four subset appears nine times; pair collection order separately shuffled.",
        comparison="Two/four task GPUs within the same four-GPU allocation, TP1 and fixed "
        "global prompt batch. Generated sequences may differ; this is total execution effect.",
        cache_policy="Fresh process/Ray namespace/scratch/compiler cache each period; shared "
        "immutable cached model/data. OS page cache, host contention and thermal carryover remain.",
        cost_scope="Historical source acquisition reported separately with zero incremental "
        "source GPU collection; CPU prediction separately. New task-provisioned GPU seconds, "
        "four-GPU invocation reservation and actual whole-allocation accounting are distinct. "
        "Failures/missing outcomes remain in costs; scheduled EndTime is never actual cost.",
    )


def freeze(root, data_root, hf_home):
    directory = root / DIRECTORY
    require(not directory.exists(), "Never overwrite an existing completion-replication freeze")
    require(not (root / INITIAL_DIRECTORY).exists(),
            "Archived initial freeze exists; use the explicit refreeze-implementation command")
    raw = data_root / "output" / GROUP
    require(not raw.exists(), "Cannot freeze after this new panel has raw records")
    runtime = runtime_hashes(root)  # Fail on missing/patched dependencies before model hashing.
    rows, targets = planned(root, data_root)
    source, fitted, inherited, input_hashes = original_inputs(root)
    metadata, cache = cached_metadata(hf_home, inherited)
    verify_old_predictions(root, source, metadata, fitted)
    started = time.monotonic()
    predictions = predict([{"settings": r} for r in targets], source, metadata, fitted)
    seconds = time.monotonic() - started
    protocol = specification()
    protocol.update(
        specified_at_utc=datetime.now(timezone.utc).isoformat(), data_root=str(data_root),
        targets=targets, metadata=metadata, model_cache=cache, input_sha256=input_hashes,
        runtime_sha256=runtime,
        data_sha256=json.loads((root / "benchmark/estimation/protocol.json").read_text())["data_sha256"],
        source_cost=eb.source_cost(predictions, source), prediction_cpu_wall_seconds=seconds)
    for name, expected in protocol["data_sha256"].items():
        require(digest(data_root / name) == expected, f"Changed original workload data: {name}")
    require(digest(data_root / "memory_tuner/code_shape_reward.py")
            == digest(root / "memory_tuner/code_shape_reward.py"), "Canonical code reward differs")
    verify_runtime(root, runtime)
    directory.mkdir()
    with (directory / "matrix.csv").open("x", newline="") as handle:
        handle.write(csv_text(rows))
        handle.flush()
        os.fsync(handle.fileno())
    write_new_json(directory / "protocol.json", protocol)
    write_new_json(directory / "predictions.json", predictions)
    seal = dict(schema_version=VERSION, status="frozen_before_new_completion_replication_execution",
                frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                files_sha256={n: digest(directory / n) for n in FREEZE_FILES})
    write_new_json(directory / "freeze.json", seal)
    sync_directory(directory)
    return dict(directory=str(directory), freeze_sha256=digest(directory / "freeze.json"),
                prediction_rows=len(predictions), **{k: protocol[k] for k in
                ("planned_pairs", "planned_processes", "target_configurations", "resource_cap")})


def archived_initial(root):
    """Authenticate the four historical files, without applying current source hashes to them."""
    directory = contained_runtime_path(root, INITIAL_DIRECTORY, directory=True)
    require({p.name for p in directory.iterdir()} == {"freeze.json", *FREEZE_FILES},
            "Archived initial freeze must contain exactly its four unchanged files")
    paths = {name: contained_runtime_path(root, INITIAL_DIRECTORY + "/" + name)
             for name in ("freeze.json", *FREEZE_FILES)}
    require(digest(paths["freeze.json"]) == INITIAL_FREEZE_SHA256,
            "Archived initial freeze anchor changed")
    seal = json.loads(paths["freeze.json"].read_text())
    require(seal["schema_version"] == VERSION
            and seal["status"] == "frozen_before_new_completion_replication_execution"
            and seal["files_sha256"] == INITIAL_FILES_SHA256, "Archived initial seal differs")
    for name, expected in INITIAL_FILES_SHA256.items():
        require(digest(paths[name]) == expected, f"Archived initial file changed: {name}")
    protocol = json.loads(paths["protocol.json"].read_text())
    require("implementation_refreeze" not in protocol, "Only one initial implementation refreeze is supported")
    return protocol, seal, {name: paths[name].read_bytes() for name in ("matrix.csv", "predictions.json")}


def retention_provenance(initial, initial_seal, runtime):
    previous = initial["runtime_sha256"]
    require(set(previous).issubset(runtime), "Implementation refreeze removed previously sealed runtime files")
    changed = sorted(name for name in previous if previous[name] != runtime[name])
    require(set(changed) <= RETENTION_IMPLEMENTATION_FILES,
            "Implementation refreeze changed runtime beyond the protocol, retention launcher and separate diagnostic")
    require("run_grpo_instrumented.slurm" in changed, "Retention launcher correction is not present")
    return dict(
        initial_directory=INITIAL_DIRECTORY, initial_freeze_sha256=INITIAL_FREEZE_SHA256,
        initial_files_sha256=INITIAL_FILES_SHA256,
        initial_frozen_at_utc=initial_seal["frozen_at_utc"], before_evaluation=True,
        changed_runtime_files=changed, added_runtime_files=sorted(set(runtime) - set(previous)),
        scope="Only Ray worker-log retention implementation and its source/execution seal are "
        "superseded before evaluation. Correct RAY_TMPDIR nesting and container session_latest "
        "symlink handling after separate diagnostics. Initial matrix, forecasts, scientific settings, "
        "data and label rules stay unchanged. Diagnostic outcomes never enter fitting or settings.")


def verify_implementation_refreeze(root, directory, protocol, seal):
    initial, initial_seal, payloads = archived_initial(root)
    scientific = {k: v for k, v in protocol.items()
                  if k not in ("runtime_sha256", "implementation_refreeze")}
    require(scientific == {k: v for k, v in initial.items() if k != "runtime_sha256"},
            "Implementation refreeze changed the initial scientific design or prediction provenance")
    require(protocol.get("implementation_refreeze") ==
            retention_provenance(initial, initial_seal, protocol["runtime_sha256"]),
            "Implementation refreeze provenance differs from the anchored initial freeze")
    for name, payload in payloads.items():
        path = contained_runtime_path(directory, name)
        require(path.read_bytes() == payload and seal["files_sha256"][name] == INITIAL_FILES_SHA256[name],
                f"Implementation refreeze changed matrix/forecast bytes: {name}")
    when, before = (datetime.fromisoformat(s["frozen_at_utc"]) for s in (seal, initial_seal))
    require(when.tzinfo is not None and before < when <= datetime.now(timezone.utc),
            "Implementation refreeze must follow the initial freeze and precede evaluation")


def refreeze_implementation(root, data_root):
    """Parent invokes once after archiving the initial directory and correcting the launcher."""
    directory = root / DIRECTORY
    require(not directory.exists(), "Archive the initial four-file directory before implementation refreeze")
    initial, initial_seal, payloads = archived_initial(root)
    require(str(data_root) == initial["data_root"], "Implementation refreeze changed the canonical data/raw root")
    raw = data_root / "output" / GROUP
    require(not raw.exists(), "Cannot refreeze implementation after evaluation claims or outputs exist")
    require(all(initial.get(k) == v for k, v in specification().items()),
            "Current implementation changed the initial scientific specification")
    original_inputs(root)  # Verify the original 90-fit inputs; never fit or collect diagnostic outcomes.
    runtime = runtime_hashes(root)
    protocol = dict(initial, runtime_sha256=runtime,
                    implementation_refreeze=retention_provenance(initial, initial_seal, runtime))
    verify_runtime(root, runtime)
    require(not raw.exists(), "Evaluation began during implementation refreeze preparation")
    directory.mkdir()
    for name, payload in payloads.items():
        with (directory / name).open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    write_new_json(directory / "protocol.json", protocol)
    seal = dict(schema_version=VERSION, status="frozen_before_new_completion_replication_execution",
                frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                files_sha256={name: digest(directory / name) for name in FREEZE_FILES})
    verify_implementation_refreeze(root, directory, protocol, seal)
    write_new_json(directory / "freeze.json", seal)
    sync_directory(directory)
    return dict(directory=str(directory), freeze_sha256=digest(directory / "freeze.json"),
                initial_freeze_sha256=INITIAL_FREEZE_SHA256,
                matrix_sha256=seal["files_sha256"]["matrix.csv"],
                predictions_sha256=seal["files_sha256"]["predictions.json"],
                changed_runtime_files=protocol["implementation_refreeze"]["changed_runtime_files"])


def _frozen_content(root, directory, freeze_sha256):
    """Scientific reconstruction; callers separately authenticate the execution closure."""
    for name in ("freeze.json", *FREEZE_FILES):
        contained_runtime_path(root, (directory / name).relative_to(root).as_posix())
    require(re.fullmatch(r"[a-f0-9]{64}", freeze_sha256 or "") is not None,
            "Externally retained freeze SHA256 is required")
    require(digest(directory / "freeze.json") == freeze_sha256, "Freeze anchor changed")
    seal = json.loads((directory / "freeze.json").read_text())
    require(seal["schema_version"] == VERSION and set(seal["files_sha256"]) == set(FREEZE_FILES)
            and seal["status"] == "frozen_before_new_completion_replication_execution",
            "Invalid freeze inventory/status")
    for name, expected in seal["files_sha256"].items():
        require(digest(directory / name) == expected, f"Changed frozen file: {name}")
    protocol = json.loads((directory / "protocol.json").read_text())
    if "implementation_refreeze" in protocol or (root / INITIAL_DIRECTORY).exists():
        verify_implementation_refreeze(root, directory, protocol, seal)
    require(all(protocol.get(k) == v for k, v in specification().items()), "Changed fixed design")
    source, fitted, metadata, inputs = original_inputs(root)
    require(inputs == protocol["input_sha256"], "Changed inherited inputs")
    require(all(protocol["metadata"].get(m) == v for m, v in metadata.items()),
            "Changed historical numeric metadata")
    for model, spec in MODELS.items():
        require(protocol["metadata"][model]["revision"] == spec["revision"]
                and protocol["model_cache"][model]["revision"] == spec["revision"],
                "Changed new model revision")
    verify_old_predictions(root, source, protocol["metadata"], fitted)
    rows, targets = planned(root, Path(protocol["data_root"]))
    require(eb.read_csv(directory / "matrix.csv") == rows and protocol["targets"] == targets,
            "Frozen schedule/targets differ from the deterministic plan")
    predictions = json.loads((directory / "predictions.json").read_text())
    expected = predict([{"settings": r} for r in targets], source, protocol["metadata"], fitted)
    require(predictions == expected, "Predictions differ from exact inherited parameters")
    when = datetime.fromisoformat(seal["frozen_at_utc"])
    require(when.tzinfo is not None and datetime.fromisoformat(protocol["specified_at_utc"])
            <= when <= datetime.now(timezone.utc), "Invalid freeze chronology")
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
    return dict(protocol=protocol, seal=seal, study_freeze=seal, rows=rows, pairs=pairs,
                targets=targets, predictions=predictions, group=GROUP,
                protocol_relative_dir=DIRECTORY, freeze_sha256=freeze_sha256,
                protocol_sha256=seal["files_sha256"]["protocol.json"],
                matrix_sha256=seal["files_sha256"]["matrix.csv"],
                allocation_validator=validate_allocation)


def load_frozen(root, directory, freeze_sha256):
    frozen = _frozen_content(root, directory, freeze_sha256)
    verify_runtime(root, frozen["protocol"]["runtime_sha256"])
    require(root == Path(__file__).resolve().parents[1], "Use Python from the supplied source root")
    return frozen


def verify_amended_sources(original, source, frozen, runtime):
    """Fixed inventories, no imports/execution from the historical source archive."""
    previous = frozen["protocol"]["runtime_sha256"]
    require(set(runtime) == set(previous), "Runtime amendment cannot add or remove sealed source names")
    verify_runtime(original, previous)
    verify_runtime(source, runtime)
    require(source == Path(__file__).resolve().parents[1], "Use Python from the amended source root")
    changed = {name for name in previous if previous[name] != runtime[name]}
    require(changed <= AMENDMENT_RUNTIME_FILES and
            {"run_grpo_instrumented.slurm", "memory_tuner/completion_replication_protocol.py"} <= changed,
            "Runtime amendment changes files beyond the narrow retention/provenance scope")
    # Shell comments/blank lines may explain the fix; executable text must differ
    # at exactly the existing, unique find inventory redirection.
    def shell_lines(root):
        return [line for line in (root / "run_grpo_instrumented.slurm").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    old_lines, new_lines = shell_lines(original), shell_lines(source)
    needle = r'''        \( -name 'worker-*.out' -o -name 'worker-*.err' \) -print0 > "$inventory"; then'''
    require(old_lines.count(needle) == 1 and
            new_lines == [line.replace(' > "$inventory"', ' >| "$inventory"')
                          if line == needle else line for line in old_lines],
            "Launcher amendment must be only the find inventory noclobber redirection")
    # Keep every preexisting scientific/endpoint helper and constant intact.
    # Only orchestration entry points may change; new helpers implement provenance.
    mutable = {
        "memory_tuner/completion_replication_protocol.py": {"load_frozen", "main"},
        "memory_tuner/run_completion_replication.py": {"execute_pair", "main"},
        "memory_tuner/collect_completion_replication.py": {"collect", "main"},
    }
    for name, exceptions in mutable.items():
        old_tree = ast.parse((original / name).read_text())
        new_tree = ast.parse((source / name).read_text())
        new_nodes = [ast.dump(node, include_attributes=False) for node in new_tree.body]
        protected_names = {}
        for node in old_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name not in exceptions:
                    protected_names[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        protected_names[child.id] = ast.dump(node, include_attributes=False)
        for node in old_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in exceptions:
                continue
            require(ast.dump(node, include_attributes=False) in new_nodes,
                    f"Amendment changed a protected scientific/runtime definition: {name}")
        for node in new_tree.body:
            bindings = ([node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        else [child.id for child in ast.walk(node)
                              if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)]
                        if isinstance(node, (ast.Assign, ast.AnnAssign)) else [])
            require(all(key not in protected_names or
                        protected_names[key] == ast.dump(node, include_attributes=False) for key in bindings),
                    f"Amendment redefines a protected scientific/runtime name: {name}")
    # Both four-file freezes must survive verbatim in both explicit archives.
    for relative in (INITIAL_DIRECTORY, DIRECTORY):
        for name in ("freeze.json", *FREEZE_FILES):
            path = relative + "/" + name
            require(contained_runtime_path(original, path).read_bytes() ==
                    contained_runtime_path(source, path).read_bytes(),
                    f"Amendment changed historical frozen bytes: {path}")
    return sorted(changed)


def tree_inventory(root, relatives):
    """Include empty directories, reject links/special files, never resolve live fallbacks."""
    files, directories = {}, []
    for relative in relatives:
        path = root / relative
        if not path.exists() and not path.is_symlink():
            continue
        contained_runtime_path(root, relative, directory=True)
        for current, children, names in os.walk(path, followlinks=False):
            current = Path(current)
            directories.append(current.relative_to(root).as_posix())
            for child in children:
                contained_runtime_path(root, (current / child).relative_to(root).as_posix(), directory=True)
            for name in names:
                key = (current / name).relative_to(root).as_posix()
                regular = contained_runtime_path(root, key)
                files[key] = dict(sha256=digest(regular), size_bytes=regular.stat().st_size)
    return dict(files=dict(sorted(files.items())), directories=sorted(directories))


def first_pair_inventory(root, frozen, *, before_amendment=False):
    """Seal/recheck finished pair 0 without interpreting its outcome or selecting trials."""
    from memory_tuner import collect_estimation_study as core
    group = contained_runtime_path(root, "output/" + GROUP, directory=True)
    ids = list(frozen["pairs"])
    first, rows = ids[0], frozen["pairs"][ids[0]]
    expected_slots = {r["experiment_id"] for r in (rows if before_amendment else frozen["rows"])}
    for path in group.iterdir():
        require(path.name in expected_slots | {"pairs"} or
                re.fullmatch(r"concurrency-[0-2]\.lock", path.name),
                "Unplanned or pre-amendment later-slot evidence")
        contained_runtime_path(root, path.relative_to(root).as_posix(),
                               directory=path.name in expected_slots | {"pairs"})
    pairs = contained_runtime_path(root, f"output/{GROUP}/pairs", directory=True)
    require({p.name for p in pairs.iterdir()} <= ({first} if before_amendment else set(ids)),
            "Other pair claims exist before amendment or outside the fixed panel")
    for path in pairs.iterdir():
        contained_runtime_path(root, path.relative_to(root).as_posix(), directory=True)
    relative = f"output/{GROUP}/pairs/{first}"
    relatives = [relative, *(f"output/{GROUP}/{r['experiment_id']}" for r in rows)]
    observed = tree_inventory(root, relatives)
    evidence = core.Evidence(root)
    manifest = evidence.json(root / relative / "pair.json")
    events = evidence.journal(root / relative / "ledger.jsonl")
    claimed, terminal = core.one_event(events, "pair_claimed"), core.one_event(events, "pair_terminal")
    summary = evidence.json(root / relative / "pair-summary.json")
    require(events[0] == claimed and events[-1] == terminal
            and all(terminal.get(k) == v for k, v in summary.items())
            and claimed["pair_id"] == manifest["pair_id"] == first
            and manifest["rows"] == rows and manifest["no_retry"] is True
            and str(manifest["job_id"]) == FIRST_JOB_ID
            and manifest["source_root"] == EXECUTION_SOURCE_ROOT
            and manifest["source_commit"] == claimed["source_commit_requested"] == EXECUTION_SOURCE_COMMIT,
            "Pair 0 must finish under its exact original source/job before amendment")
    for item in (manifest, claimed):
        require(item["freeze_sha256"] == frozen["freeze_sha256"]
                and item["matrix_sha256"] == frozen["matrix_sha256"]
                and item["protocol_sha256"] == frozen["protocol_sha256"]
                and "runtime_amendment_sha256" not in item,
                "Pair 0 cannot be rebound to amended provenance")
    for row in rows:
        period = core.one_event(events, "terminal", int(row["period"]))
        record = evidence.json(root / relative / f"period-{row['period']}.json")
        require({k: v for k, v in period.items() if k not in {
            "event", "sequence", "timestamp_ns", "monotonic_ns"}} == record,
            "Pair 0 terminal records are not final")
        parent = root / f"output/{GROUP}/{row['experiment_id']}"
        require(not parent.exists() or {p.name for p in parent.iterdir()} == {"attempt-" + FIRST_JOB_ID},
                "Pair 0 contains a replacement or foreign attempt")
    require(datetime.fromisoformat(frozen["seal"]["frozen_at_utc"]).timestamp() * 1e9
            < claimed["timestamp_ns"] < terminal["timestamp_ns"] <= time.time_ns(),
            "Pair 0 has invalid pre-amendment chronology")
    evidence.verify_unchanged()
    require(observed == tree_inventory(root, relatives), "Pair 0 changed while inventorying")
    return dict(pair_id=first, index=0, job_id=FIRST_JOB_ID,
                claimed_at_ns=claimed["timestamp_ns"], finished_at_ns=terminal["timestamp_ns"],
                **observed)


def amendment_sources(source, original, directory, freeze_sha256, runtime):
    require(freeze_sha256 == EXECUTION_FREEZE_SHA256, "Amendment requires the exact execution freeze anchor")
    require(directory == source / DIRECTORY, "Amendment requires the unchanged canonical protocol directory")
    frozen = _frozen_content(original, original / DIRECTORY, freeze_sha256)
    changed = verify_amended_sources(original, source, frozen, runtime)
    return frozen, changed


def seal_runtime_amendment(source, original, root, directory, freeze_sha256, source_commit, output):
    """Explicit parent-only command, after pair 0 finishes and before any later claim."""
    from memory_tuner.run_completion_replication import clean_source
    require(root == CANONICAL_ROOT, "Seal against the canonical raw root")
    require(re.fullmatch(r"[a-f0-9]{40}", source_commit or "")
            and source_commit != EXECUTION_SOURCE_COMMIT, "Exact distinct amended source commit required")
    require(output.is_absolute() and ".." not in output.parts
            and output.parent.is_dir() and not any(p.is_symlink() for p in output.parents),
            "Amendment output requires an existing explicit unlinked parent directory")
    require(not output.exists() and not output.is_symlink(), "Never replace an amendment")
    require(not any(output.is_relative_to(p) for p in (source, original, root / "output")),
            "Retain the amendment separately from source and raw evidence")
    clean_source(source, source_commit)
    protocol_path = contained_runtime_path(original, DIRECTORY + "/protocol.json")
    require(digest(contained_runtime_path(original, DIRECTORY + "/freeze.json")) == freeze_sha256
            == EXECUTION_FREEZE_SHA256, "Amendment requires the exact execution freeze anchor")
    previous = json.loads(protocol_path.read_text())["runtime_sha256"]
    runtime = {name: digest(contained_runtime_path(source, name)) for name in previous}
    frozen, changed = amendment_sources(source, original, directory, freeze_sha256, runtime)
    first = first_pair_inventory(root, frozen, before_amendment=True)
    sealed_ns = time.time_ns()
    require(first["finished_at_ns"] < sealed_ns, "Pair 0 must finish before amendment")
    document = dict(
        schema_version=RUNTIME_AMENDMENT_VERSION, scope=AMENDMENT_SCOPE, sealed_at_ns=sealed_ns,
        original_freeze_sha256=freeze_sha256, initial_freeze_sha256=INITIAL_FREEZE_SHA256,
        original_files_sha256=frozen["seal"]["files_sha256"],
        original_source_root=EXECUTION_SOURCE_ROOT, original_source_commit=EXECUTION_SOURCE_COMMIT,
        original_runtime_sha256=frozen["protocol"]["runtime_sha256"],
        amended_source_root=str(source), amended_source_commit=source_commit,
        amended_runtime_sha256=runtime, changed_runtime_files=changed,
        allowed_indices=list(AMENDED_INDICES), allowed_pair_ids=list(frozen["pairs"])[1:],
        pre_amendment_pair0=first)
    require(first == first_pair_inventory(root, frozen, before_amendment=True),
            "Raw evidence changed during amendment preparation")
    clean_source(source, source_commit)
    verify_amended_sources(original, source, frozen, runtime)
    require(first == first_pair_inventory(root, frozen, before_amendment=True),
            "A later claim or changed pair 0 appeared before amendment sealing")
    # No overwrite, and never write into either frozen directory.
    write_new_json(output, document)
    return dict(path=str(output), amendment_sha256=digest(output),
                freeze_sha256=freeze_sha256, allowed_indices=list(AMENDED_INDICES))


def load_runtime_amendment(source, original, root, directory, freeze_sha256, path, anchor):
    """CPU-only, explicit old/new source roots; the external digest is mandatory."""
    require(original is not None and path is not None
            and re.fullmatch(r"[a-f0-9]{64}", anchor or ""), "Explicit amendment, original source and SHA256 required")
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
            "Amendment path must be explicit and unlinked")
    contained_runtime_path(path.parent, path.name)
    require(digest(path) == anchor, "Externally retained amendment anchor changed")
    document = json.loads(path.read_text())
    require(set(document) == {
        "schema_version", "scope", "sealed_at_ns", "original_freeze_sha256", "initial_freeze_sha256",
        "original_files_sha256", "original_source_root", "original_source_commit",
        "original_runtime_sha256", "amended_source_root", "amended_source_commit",
        "amended_runtime_sha256", "changed_runtime_files", "allowed_indices", "allowed_pair_ids",
        "pre_amendment_pair0"}, "Unknown or missing amendment claims")
    require(document["schema_version"] == RUNTIME_AMENDMENT_VERSION
            and document["scope"] == AMENDMENT_SCOPE
            and document["original_freeze_sha256"] == freeze_sha256
            and document["initial_freeze_sha256"] == INITIAL_FREEZE_SHA256
            and document["original_source_root"] == EXECUTION_SOURCE_ROOT
            and document["original_source_commit"] == EXECUTION_SOURCE_COMMIT
            and document["allowed_indices"] == list(AMENDED_INDICES)
            and all(type(i) is int for i in document["allowed_indices"]),
            "Amendment exceeds the single fixed implementation scope")
    frozen, changed = amendment_sources(source, original, directory, freeze_sha256,
                                        document["amended_runtime_sha256"])
    require(document["original_runtime_sha256"] == frozen["protocol"]["runtime_sha256"]
            and document["original_files_sha256"] == frozen["seal"]["files_sha256"]
            and document["changed_runtime_files"] == changed
            and document["allowed_pair_ids"] == list(frozen["pairs"])[1:],
            "Amendment source, forecast or allowed-pair binding differs")
    require(re.fullmatch(r"[a-f0-9]{40}", document["amended_source_commit"])
            and document["amended_source_commit"] != EXECUTION_SOURCE_COMMIT
            and Path(document["amended_source_root"]).is_absolute()
            and ".." not in Path(document["amended_source_root"]).parts,
            "Invalid amended execution source identity")
    first = first_pair_inventory(root, frozen)
    require(first == document["pre_amendment_pair0"], "Sealed pre-amendment pair 0 evidence changed")
    require(type(document["sealed_at_ns"]) is int
            and first["finished_at_ns"] < document["sealed_at_ns"] <= time.time_ns(),
            "Amendment must follow finished pair 0 and precede later claims")
    require(digest(path) == anchor, "Amendment changed during verification")
    return dict(frozen, runtime_amendment=document, runtime_amendment_sha256=anchor)


def pair_runtime(frozen, index):
    """Effective source boundary is explicit, never relabel the old scientific freeze."""
    amendment = frozen.get("runtime_amendment")
    if not amendment or index == 0:
        return dict(frozen, runtime_amendment_sha256=None)
    require(index in AMENDED_INDICES, "Runtime amendment does not authorize this index")
    return dict(frozen, study_freeze={
        "frozen_at_utc": datetime.fromtimestamp(amendment["sealed_at_ns"] / 1e9, timezone.utc).isoformat()})


def validate_runtime_binding(evidence, frozen, pair, index):
    """Validate source/claim/dispatch bindings, independently of endpoint classification."""
    amendment = frozen.get("runtime_amendment")
    if amendment is None:
        return
    manifest, events = pair["manifest"], pair["events"]
    claimed = next(e for e in events if e["event"] == "pair_claimed")
    amended = index in AMENDED_INDICES
    expected_source = amendment["amended_source_root"] if amended else EXECUTION_SOURCE_ROOT
    expected_commit = amendment["amended_source_commit"] if amended else EXECUTION_SOURCE_COMMIT
    expected_anchor = frozen["runtime_amendment_sha256"] if amended else None
    require(manifest["source_root"] == expected_source
            and claimed["source_commit_requested"] == expected_commit
            and manifest.get("source_commit") in (expected_commit, None),
            "Pair source differs from its original/amended index binding")
    require(claimed["timestamp_ns"] > amendment["sealed_at_ns"] if amended else
            events[-1]["timestamp_ns"] < amendment["sealed_at_ns"], "Pair crosses its amendment chronology")
    if amended and manifest["allocation"] is not None:
        require(amendment["sealed_at_ns"] < manifest["allocation"]["started_at_epoch"] * 1e9
                <= claimed["timestamp_ns"], "Amended allocation must start after the amendment")
    for item in (manifest, claimed):
        require(item.get("runtime_amendment_sha256") == expected_anchor
                and (amended or "runtime_amendment_sha256" not in item),
                "Pair is bound to the wrong runtime amendment")
        if amended:
            require(item.get("pair_index") == index, "Amended pair index differs")
    for row in frozen["pairs"][manifest["pair_id"]]:
        attempt = evidence.root / f"output/{GROUP}/{row['experiment_id']}/attempt-{manifest['job_id']}"
        dispatch = attempt / "dispatch.json"
        if dispatch.exists():
            item = evidence.json(dispatch)
            require(item.get("runtime_amendment_sha256") == expected_anchor
                    and (amended or "runtime_amendment_sha256" not in item),
                    "Dispatch is bound to the wrong runtime amendment")
            if amended:
                require(item.get("pair_index") == index, "Amended dispatch index differs")


def amendment_arguments(parser):
    """Environment equivalents let the unchanged Slurm script forward explicit authority."""
    parser.add_argument("--runtime-amendment", type=Path,
                        default=os.environ.get("COMPLETION_RUNTIME_AMENDMENT"))
    parser.add_argument("--amendment-sha256", default=os.environ.get("COMPLETION_AMENDMENT_SHA256"))
    parser.add_argument("--original-source-root", type=Path,
                        default=os.environ.get("COMPLETION_ORIGINAL_SOURCE_ROOT"))


def selected_runtime(source, directory, root, freeze_sha256, path=None, anchor=None, original=None):
    if path is None and anchor is None and original is None:
        return load_frozen(source, directory, freeze_sha256)
    require(original is not None, "Explicit original source root required; no live-path fallback")
    return load_runtime_amendment(source, Path(original).resolve(), root, directory,
                                  freeze_sha256, path, anchor)


def validate_allocation(allocation):
    from memory_tuner import run_matched_gpu as runner
    runner.validate_allocation_inventory(allocation["gpus"])
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_/]*)=([^\s]+)",
                             allocation["scheduler_record"]))
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    require(tres.get("gres/gpu") == "4" and tres.get("cpu") == "64"
            and tres.get("mem") in {"384G", "393216M"} and fields.get("CPUs/Task") == "64"
            and fields.get("Requeue") == "0" and fields.get("Restarts", "0") == "0",
            "Require exactly four GPUs, 64 CPUs, 384 GiB host memory and no requeue/restart")
    require(0 < allocation["ends_at_epoch"] - allocation["started_at_epoch"] <= 3301,
            "Allocation exceeds 55 minutes")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "freeze", "refreeze-implementation",
                                            "seal-runtime-amendment"),
                        nargs="?", default="preview")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--full", action="store_true", help="Include all target, order and prediction rows")
    parser.add_argument("--original-source-root", type=Path)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root, data = args.source_root.resolve(), args.data_root.resolve()
    require(data == CANONICAL_ROOT, "Use the canonical data/raw root")
    if args.command == "seal-runtime-amendment":
        require(args.original_source_root is not None and args.output is not None,
                "--original-source-root and --output are required for the explicit amendment")
        result = seal_runtime_amendment(root, args.original_source_root.resolve(), data,
                                       root / DIRECTORY, args.freeze_sha256,
                                       args.source_commit, args.output.absolute())
    elif args.command == "refreeze-implementation":
        result = refreeze_implementation(root, data)
    elif args.command == "freeze":
        require(args.hf_home is not None, "--hf-home is required for an explicit freeze")
        result = freeze(root, data, args.hf_home.resolve())
    else:
        rows, targets = planned(root, data)
        source, fitted, metadata, _ = original_inputs(root)
        verify_old_predictions(root, source, metadata, fitted)
        result = dict(specification(), writes_performed=False, old_qwen_predictions_exact=True,
                      first_task_gpu_counts=dict(Counter(r["gpu_count"] for r in rows
                                                         if r["period"] == "1")),
                      two_gpu_subsets=dict(Counter(r["gpu_subset_ordinals"] for r in rows
                                                   if r["gpu_count"] == "2")))
        if args.hf_home:
            metadata, _ = cached_metadata(args.hf_home.resolve(), metadata, hash_payloads=False)
            predictions = predict([{"settings": r} for r in targets], source, metadata, fitted)
            result["numeric_metadata"] = {m: {k: metadata[m][k] for k in (
                "revision", "floating_checkpoint_elements", "all_linear_layer_adapter_elements",
                "num_hidden_layers", "hidden_size", "vocab_size")} for m in MODELS}
            result["prediction_counts"] = {
                model: {method: dict(Counter(p["predicted_state"] for p in predictions
                    if p["method"] == method and p["configuration_id"] in {
                        t["configuration_id"] for t in targets if t["model_id"] == model}))
                    for method in eb.METHODS} for model in MODELS}
            if args.full:
                result["predictions"] = predictions
        if args.full:
            result.update(targets=targets, pair_order=list(dict.fromkeys(r["pair_id"] for r in rows)))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
