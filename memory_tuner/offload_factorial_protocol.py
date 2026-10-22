"""Independent offload/batch/reservation intervention; preview then immutable freeze.

This outcome-motivated panel is separate from every earlier cohort. No fitting,
downloads, submissions, retries, or historical mutations occur in this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import random
import re

import numpy as np

from memory_tuner import completion_replication_protocol as inherited
from memory_tuner import estimation_baselines as eb
from memory_tuner.device_contract import digest, sync_directory, write_new_json

GROUP = "offload_factorial"
DIRECTORY = "benchmark/offload_factorial"
VERSION = "offload-factorial-1"
CANONICAL_ROOT = Path("/mnt/proj3/open-35-44/verl/rl")
MODELS = inherited.MODELS
DATA_ROOTS = inherited.DATA_ROOTS
DATASETS = inherited.DATASETS
SEEDS = (241, 242, 243)
BATCHES = (4, 8)
RESERVATIONS = (.45, .70)
OFFLOADS = ("False", "True")
ORDER_SEEDS = (2026091741, 2026091742)
CAPS = dict(
    max_concurrent_allocations=3, allocations=18, gpus_per_allocation=4,
    cpus_per_task=64, host_memory_mib=393216, wall_seconds=10800,
    payload_timeout_seconds=900, idle_timeout_seconds=180,
    cleanup_reserve_seconds=60, maximum_reserved_gpu_hours=216,
    retries=0, requeue=False, replacement=False,
)
METHODS = ("frozen_logistic", "component_regression", "offload_only",
           "strict_c1", "reservation_only", "offload_or_low_reservation")
INHERITED_FREEZE = "ba3e24e9e3a13b60ca35b22f92ec60d5dac322a80d8f68e660bfd3e98ef1f922"
INHERITED_PROTOCOL = "1eb79c4cf75f1292e72b4e1253e2d6d569f9b6fe41602a8a72bf6914467a0683"
FREEZE_FILES = ("protocol.json", "matrix.csv", "predictions.json")
require = inherited.require


def inherited_inputs(root):
    root = Path(root)
    directory = root / "benchmark/completion_replication"
    require(digest(directory / "freeze.json") == INHERITED_FREEZE,
            "Inherited source panel seal changed")
    require(digest(directory / "protocol.json") == INHERITED_PROTOCOL,
            "Inherited source metadata/cache specification changed")
    seal = json.loads((directory / "freeze.json").read_text())
    require(seal["files_sha256"]["protocol.json"] == INHERITED_PROTOCOL,
            "Inherited protocol is not bound to its seal")
    original = json.loads((directory / "protocol.json").read_text())
    source, fitted, metadata, hashes = inherited.original_inputs(root)
    require(all(original["metadata"][k] == v for k, v in metadata.items()),
            "Original architecture metadata changed")
    hashes.update({
        "benchmark/completion_replication/freeze.json": INHERITED_FREEZE,
        "benchmark/completion_replication/protocol.json": INHERITED_PROTOCOL,
    })
    return source, fitted, original, hashes


def condition_id(offload, batch, reservation):
    return f"o{int(offload == 'True')}-b{batch}-r{round(100 * reservation)}"


def anchor(offload, batch, reservation):
    return ((offload == "True" and batch == 4 and reservation == .45)
            or (offload == "False" and batch == 8 and reservation == .60))


def planned(root, data_root=CANONICAL_ROOT):
    """Generate the complete fixed grid without reading new target outcomes."""
    source, _, _, _ = inherited_inputs(root)
    old_rows = eb.read_csv(Path(root) / "benchmark/estimation/matrix.csv")
    factors = list(itertools.product(OFFLOADS, BATCHES, RESERVATIONS))
    # Eight-treatment Williams sequences: every ordered adjacent pair occurs
    # once in the eight-sequence set. 18 blocks use two full sets plus two
    # distinct sequences. Realized balance is reported, never assumed exact.
    first = (0, 1, 7, 2, 6, 3, 5, 4)
    sequences = [tuple((value + shift) % 8 for value in first) for shift in range(8)]
    schedule = list(range(8)) * 2 + [0, 4]
    blocks = list(itertools.product(MODELS, DATASETS, SEEDS))
    random.Random(ORDER_SEEDS[1]).shuffle(blocks)
    order = random.Random(ORDER_SEEDS[0])
    for _ in range(10000):
        order.shuffle(schedule)
        if all(len({schedule[i] for i, block in enumerate(blocks) if block[:2] == case}) == 3
               for case in itertools.product(MODELS, DATASETS)):
            break
    else:
        raise ValueError("Could not construct the predeclared sequence schedule")
    rows, targets = [], []
    for model, dataset in itertools.product(MODELS, DATASETS):
        spec = MODELS[model]
        template = dict(next(r["settings"] for r in source
                             if r["settings"]["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
                             and r["settings"]["legacy_dataset_id"] == dataset
                             and r["settings"]["configuration_level"] == "c1"
                             and r["settings"]["gpu_count"] == "2"))
        for offload, batch, reservation in factors:
            condition = condition_id(offload, batch, reservation)
            target = dict(
                template, configuration_id=f"of-{spec['slug']}-{dataset}-{condition}",
                case_id=f"of-{spec['slug']}-{dataset}", configuration_level=condition,
                study=GROUP, model_id=model, model_family=spec["family"],
                model_name=model.split("/")[-1], gpu_count="4",
                actor_micro_batch=str(batch), vllm_gpu_memory_utilization=str(reservation),
                parameter_offload=offload, optimizer_offload="False",
                free_cache_engine="True", total_training_steps="1",
                allocator_trace_enabled="True", allocator_trace_sync="1",
                save_freq="-1", test_freq="-1", val_before_train="False",
                resume_mode="disable", max_actor_ckpt_to_keep="1",
                condition=condition, novelty="anchor" if anchor(offload, batch, reservation)
                else "challenge",
            )
            targets.append(target)
    for block_index, (model, dataset, seed) in enumerate(blocks):
        spec = MODELS[model]
        block = f"of-{spec['slug']}-{dataset}-s{seed}"
        for period, factor_index in enumerate(sequences[schedule[block_index]], 1):
            offload, batch, reservation = factors[factor_index]
            condition = condition_id(offload, batch, reservation)
            target = next(t for t in targets if t["model_id"] == model
                          and t["legacy_dataset_id"] == dataset and t["condition"] == condition)
            row = dict(next(r for r in old_rows
                            if r["dataset"] == dataset and r["gpu_count"] == "4"))
            for key in ("actor_micro_batch", "vllm_gpu_memory_utilization",
                        "parameter_offload", "optimizer_offload", "free_cache_engine",
                        "allocator_trace_enabled", "allocator_trace_sync", "save_freq",
                        "test_freq", "val_before_train", "resume_mode", "max_actor_ckpt_to_keep",
                        "total_training_steps", "configuration_id", "case_id",
                        "configuration_level", "novelty"):
                row[key] = target[key]
            row.update(
                experiment_id=f"{block}-{condition}", block_id=block, pair_id=block,
                block_index=str(block_index), sequence_id=str(schedule[block_index]),
                period=str(period), condition=condition, run_group=GROUP,
                study_stage=GROUP, revision_study=GROUP, model=model,
                model_family=spec["family"], model_revision=spec["revision"],
                training_seed=str(seed), data_dir=str(Path(data_root) / DATA_ROOTS[dataset]),
                gpu_subset_ordinals="0;1;2;3", per_process_timeout_seconds="900",
                custom_reward_function_path=str(Path(data_root) / "memory_tuner/code_shape_reward.py")
                if dataset == "code_heavy_tail" else "null",
                custom_reward_function_name="compute_score", capture_response_lengths="False",
            )
            rows.append(row)
    validate_schedule(rows, targets)
    return rows, targets


def validate_schedule(rows, targets):
    require(len(rows) == 144 and len(targets) == 48, "Incomplete independent-factor grid")
    require(len({r["experiment_id"] for r in rows}) == 144, "Duplicate experiment")
    require(len({t["configuration_id"] for t in targets}) == 48, "Duplicate configuration")
    require(Counter(t["novelty"] for t in targets) == {"anchor": 6, "challenge": 42},
            "Wrong anchor/challenge split")
    cells = set(itertools.product(OFFLOADS, map(str, BATCHES), map(str, RESERVATIONS)))
    blocks = set(r["block_id"] for r in rows)
    require(len(blocks) == 18, "Wrong block count")
    for block in blocks:
        part = [r for r in rows if r["block_id"] == block]
        require({int(r["period"]) for r in part} == set(range(1, 9)),
                "Missing/duplicate block period")
        require({(r["parameter_offload"], r["actor_micro_batch"],
                  r["vllm_gpu_memory_utilization"]) for r in part} == cells,
                "Factors are not fully crossed within block")
        require(len({(r["model"], r["dataset"], r["training_seed"], r["case_id"])
                     for r in part}) == 1, "Block changes model, workload, or seed")
        require(all(r["gpu_count"] == "4" and r["gpu_subset_ordinals"] == "0;1;2;3"
                    and r["pair_id"] == block for r in part), "Changed block device mapping")
    for target in targets:
        part = [r for r in rows if r["configuration_id"] == target["configuration_id"]]
        require(len(part) == 3 and {int(r["training_seed"]) for r in part} == set(SEEDS),
                "Configuration lacks its three distinct seeds")
    position = Counter((r["condition"], r["period"]) for r in rows)
    require(set(position.values()) <= {2, 3} and len(position) == 64,
            "Position near-balance failed")
    for case in {r["case_id"] for r in rows}:
        require(len({r["sequence_id"] for r in rows if r["case_id"] == case}) == 3,
                "A case must receive three distinct condition-order sequences")
    adjacent = Counter()
    for block in blocks:
        part = sorted((r for r in rows if r["block_id"] == block), key=lambda r: int(r["period"]))
        adjacent.update((a["condition"], b["condition"]) for a, b in zip(part, part[1:]))
    require(len(adjacent) == 56 and set(adjacent.values()) <= {2, 3},
            "Adjacent-condition near-balance failed")


def predictions(root, targets):
    """Reuse old fitted weights; newly written shortcut rules are admission-only."""
    _, fitted, original, _ = inherited_inputs(root)
    records = [{"settings": t} for t in targets]
    require(fitted["feature_names"] == list(eb.FEATURES), "Frozen feature order changed")
    x = eb.features(records, original["metadata"])
    regression = np.column_stack((np.ones(len(targets)),
                                  x / np.asarray(fitted["component_divisors"])))
    peaks = regression @ np.asarray(fitted["component_coefficients_gib"]) * 1024
    scaled = (x - np.asarray(fitted["scaler_mean"])) / np.asarray(fitted["scaler_scale"])
    logits = scaled @ np.asarray(fitted["classifier_coefficients"]).T + fitted["classifier_intercepts"]
    classes = [fitted["classifier_classes"][i] for i in np.argmax(logits, axis=1)]
    output = []
    for index, target in enumerate(targets):
        startup = eb.startup_diagnostic(target, original["metadata"])
        peak_state = eb.peak_state(float(peaks[index]), target)
        regression_state = "memory_failure" if startup["predicted_startup_budget_failure"] else peak_state
        for method, state, peak in (
            ("frozen_logistic", classes[index], None),
            ("component_regression", regression_state, float(peaks[index])),
        ):
            output.append(dict(
                configuration_id=target["configuration_id"], method=method,
                prediction_type="three_state", predicted_state=state,
                predicted_peak_mib=peak, approved=state == "within_margin",
                startup_guard_changed_state=(regression_state != peak_state)
                if method == "component_regression" else None,
            ))
        offload = target["parameter_offload"] == "True"
        low_reservation = float(target["vllm_gpu_memory_utilization"]) == .45
        rules = dict(offload_only=offload,
                     strict_c1=offload and low_reservation and target["actor_micro_batch"] == "4",
                     reservation_only=low_reservation,
                     offload_or_low_reservation=offload or low_reservation)
        for method, approved in rules.items():
            output.append(dict(
                configuration_id=target["configuration_id"], method=method,
                prediction_type="admission_only", predicted_state=None,
                predicted_peak_mib=None, approved=approved, startup_guard_changed_state=None,
            ))
    return output


def prediction_contract(targets, forecasts):
    indexed = {(p["configuration_id"], p["method"]): p for p in forecasts}
    difference = []
    for target in targets:
        cid = target["configuration_id"]
        classifier = indexed[cid, "frozen_logistic"]["approved"]
        shortcut = indexed[cid, "offload_or_low_reservation"]["approved"]
        require(not classifier or shortcut, "Frozen subset relation changed")
        if shortcut and not classifier:
            difference.append(dict(
                configuration_id=cid, model=target["model_id"],
                workload=target["legacy_dataset_id"], condition=target["condition"],
            ))
    require(len(difference) == 7, "Expected seven prediction-selected comparisons")
    return dict(
        selected_before_target_outcomes=True,
        comparison="combined-shortcut approvals rejected by the classifier",
        disagreements=difference,
        model_counts=dict(Counter(r["model"] for r in difference)),
        approved_configurations={m: sum(p["approved"] for p in forecasts if p["method"] == m)
                                 for m in METHODS},
        unit="configuration with a three-seed assessment, not individual seed",
    )


def specification():
    return dict(
        schema_version=VERSION, group=GROUP, allocation_gpus=4, models=MODELS,
        datasets=list(DATASETS), seeds=list(SEEDS), batches=list(BATCHES),
        reservations=list(RESERVATIONS), parameter_offload=list(OFFLOADS),
        planned_blocks=18, planned_processes=144, target_configurations=48,
        scheduler_record_timezone="UTC",
        resource_cap=CAPS, device_capacity_mib=40960, memory_margin_limit_mib=38912,
        methods=list(METHODS), inherited_prediction_seal=inherited.ORIGINAL_SEAL_SHA256,
        interpretation="Outcome-motivated, prospective independent-factor intervention in familiar "
        "models/workloads. Previous outcomes informed this design and shortcut rules. New target "
        "outcomes do not enter selection, prediction, fitting or thresholds. This is not an "
        "unseen-family or cross-hardware validation.",
        contrast="Only actor-parameter offload changes within each fixed model/workload/seed/"
        "batch/reservation contrast on the same four GPUs. Generated responses can still differ; "
        "this estimates total execution effects, not tensor-identical isolated allocator effects.",
        prediction="Original 90-configuration logistic/component fits reused verbatim without "
        "refitting. No newly adapted donor method. Report familiar anchors separately from the "
        "42 independently crossed settings and show each model/offload group. Only six "
        "exact original-c1 anchors occur; original c2 at reservation.60 is not in this grid.",
        shortcuts="offload_only approves offload=True; strict_c1 additionally requires batch=4 "
        "and reservation=.45; reservation_only approves reservation=.45; "
        "offload_or_low_reservation approves offload=True OR reservation=.45. "
        "strict_c1 is an anchor-only negative control, not a competitive challenge predictor. "
        "These are admission-only "
        "rules, not invented three-class label predictors. GPU count is fixed at four.",
        design_selection="The prior .45/.60 grid was inspected using frozen predictions only. "
        "Before new GPU execution, .45/.70 was chosen to challenge the apparent offload shortcut: "
        "classifier/combined-rule disagreement increases from three to seven novel settings "
        "and involves both model families. This is targeted design, not representative sampling "
        "or optimization against new outcomes. Prior resident.70 failures used batch16 and "
        "do not establish the new batch4/8 outcomes.",
        primary="Failure-inclusive paired offload outcome transitions at each fixed batch/reservation; "
        "configuration-level useful approvals, failure approvals, above-margin approvals and "
        "within-margin rejections on challenge configurations, including unresolved coverage. "
        "The classifier's 29 approvals are a subset of the combined shortcut's36, so its "
        "primary predictive question is erroneous approvals avoided versus usable configurations "
        "lost among the seven disagreements, not greater usable-configuration recovery.",
        secondary="Anchor results and whole-panel scores; continuous stage/whole-run contrasts "
        "conditional on matched completion; report complete-pair/block counts, equal-case means "
        "only when all six cases contribute, within-case ranges and interactions. Pooled continuous "
        "summaries average observed matched completions, not the complete factorial grid: an "
        "entire failing stratum can disappear. Fixed batch/reservation strata remain primary; "
        "empty continuous strata are not estimable. The within-margin indicator measures "
        "completion AND margin compliance, not completion alone.",
        uncertainty="Selected cases and related grid cells are not independent population draws. "
        "Seeds assess repeatability. Main summaries show all case effects and observed within-case "
        "ranges; no naive binomial confidence intervals over configurations or seed replication.",
        outcome_rule="All three planned distinct seeds must have validated completed or diagnosed "
        "memory-failure outcomes. Any failure then defines memory_failure; otherwise any completed "
        "peak above38912MiB defines above_margin; otherwise within_margin. Missing/invalid/timeout "
        "slots leave configuration unresolved; preserve partial known-failure/margin flags.",
        failure_policy="One permanent block claim; no retry, requeue, replacement or data-dependent "
        "extension. Stop remaining launches after failed cleanup/idle or inadequate full timeout "
        "budget. All eight planned slots remain in accounting regardless of launch or eligibility.",
        launch_gate="Run frozen block0 first; release remaining indices1-17 only after independent "
        "raw reconstruction validates its eight completed or diagnosed-memory-failure outcomes, "
        "all device/source/cleanup evidence and terminal allocation. Gate uses implementation "
        "validity only, not memory-label composition or predictor performance. Retain block0 "
        "in every analysis; preserve any invalid attempt and explicitly amend before continuing.",
        randomization="18 shuffled blocks use two full eight-treatment Williams sequence sets plus "
        "two distinct additional sequences. Each condition occurs two or three times per position; "
        "every ordered consecutive-condition pair occurs two or three times. Each case receives "
        "three distinct sequences. Global and per-case realized order are reported, not described "
        "as exact within-case balance.",
        success_interpretation="Avoiding erroneous approvals without losing useful settings supports "
        "an advantage over this specific combined rule on this selected panel, only if no "
        "still-unknown disagreement could reverse the zero-loss conclusion. Avoiding errors while "
        "losing useful settings establishes a tradeoff, not unconditional superiority. Partial "
        "validated failure/exceedance can establish known unsuitability without resolving a "
        "three-seed label; report it separately. Unknown disagreements can change conclusions. "
        "Ties, shortcut wins, failure-dominated or unresolved panels are valid findings and "
        "will narrow claims; no outcome is required for publication.",
        cache_policy="Fresh process/Ray namespace/scratch/compiler cache per condition; shared "
        "immutable model/data snapshots. Host contention, OS cache and thermal carryover remain.",
        cost_scope="Whole outer scheduler allocation GPU-seconds, including setup/failures/idle, "
        "reported separately from invocation GPU-seconds; never add these overlapping views.",
    )


def freeze(root, data_root=CANONICAL_ROOT):
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    directory = root / DIRECTORY
    require(not directory.exists() and not (data_root / "output" / GROUP).exists(),
            "Never freeze after creating this panel's inputs or raw claims")
    require(all(os.environ.get(name) == "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "Freeze requires OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1")
    rows, targets = planned(root, data_root)
    _, _, original, hashes = inherited_inputs(root)
    runtime = inherited.runtime_hashes(root)
    needed = {f"memory_tuner/{name}" for name in (
        "offload_factorial_protocol.py", "offload_factorial_runtime.py",
        "run_offload_factorial.py", "collect_offload_factorial.py",
        "offload_factorial_analysis.py")}
    require(needed <= runtime.keys(), "Missing new execution/analysis source")
    runtime["memory_tuner/run_offload_factorial.slurm"] = digest(
        root / "memory_tuner/run_offload_factorial.slurm")
    for name, expected in original["data_sha256"].items():
        require(digest(data_root / name) == expected, "Changed inherited workload data")
    for model in MODELS:
        inherited.validate_cache(original, model)
    forecasts = predictions(root, targets)
    protocol = specification()
    protocol.update(
        specified_at_utc=datetime.now(timezone.utc).isoformat(), data_root=str(data_root),
        targets=targets, input_sha256=hashes, runtime_sha256=runtime,
        metadata=original["metadata"], model_cache=original["model_cache"],
        data_sha256=original["data_sha256"],
        primary_prediction_comparison=prediction_contract(targets, forecasts),
    )
    directory.mkdir()
    with (directory / "matrix.csv").open("x", newline="") as handle:
        handle.write(inherited.csv_text(rows))
        handle.flush()
        os.fsync(handle.fileno())
    write_new_json(directory / "protocol.json", protocol)
    write_new_json(directory / "predictions.json", forecasts)
    write_new_json(directory / "freeze.json", dict(
        schema_version=VERSION, status="sealed_before_offload_factorial_execution",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        files_sha256={name: digest(directory / name) for name in FREEZE_FILES},
    ))
    sync_directory(directory)
    return dict(directory=str(directory), freeze_sha256=digest(directory / "freeze.json"),
                planned_processes=144, planned_blocks=18, target_configurations=48,
                prediction_rows=len(forecasts), resource_cap=CAPS)


def load(root, freeze_sha256, verify_assets=False):
    root = Path(root).resolve()
    directory = root / DIRECTORY
    require(re.fullmatch(r"[0-9a-f]{64}", freeze_sha256 or "") is not None
            and digest(directory / "freeze.json") == freeze_sha256, "Wrong external freeze anchor")
    require(all(os.environ.get(name) == "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "Replay requires OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1")
    seal = json.loads((directory / "freeze.json").read_text())
    require(seal["schema_version"] == VERSION
            and seal["status"] == "sealed_before_offload_factorial_execution"
            and set(seal["files_sha256"]) == set(FREEZE_FILES), "Malformed freeze")
    stamp = datetime.fromisoformat(seal["frozen_at_utc"])
    require(stamp.utcoffset() is not None and stamp < datetime.now(timezone.utc),
            "Freeze must have an explicit past UTC timestamp")
    for name, expected in seal["files_sha256"].items():
        require(digest(directory / name) == expected, "Changed frozen input: " + name)
    protocol = json.loads((directory / "protocol.json").read_text())
    require(all(protocol.get(k) == v for k, v in specification().items()),
            "Frozen scientific specification differs")
    rows, targets = planned(root, Path(protocol["data_root"]))
    require(rows == eb.read_csv(directory / "matrix.csv") and targets == protocol["targets"],
            "Frozen complete factorial matrix differs")
    forecasts = json.loads((directory / "predictions.json").read_text())
    require(forecasts == predictions(root, targets), "Frozen prediction replay differs")
    require(protocol["primary_prediction_comparison"] == prediction_contract(targets, forecasts),
            "Frozen primary prediction comparison differs")
    inherited.verify_runtime(root, protocol["runtime_sha256"])
    for name, expected in protocol["input_sha256"].items():
        require(digest(root / name) == expected, "Changed inherited fitting input")
    if verify_assets:
        for name, expected in protocol["data_sha256"].items():
            require(digest(Path(protocol["data_root"]) / name) == expected, "Changed workload asset")
        for model in MODELS:
            inherited.validate_cache(protocol, model)
    return dict(
        rows=rows, protocol=protocol, predictions=forecasts, seal=seal, freeze_sha256=freeze_sha256,
        matrix_sha256=seal["files_sha256"]["matrix.csv"],
        protocol_sha256=seal["files_sha256"]["protocol.json"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "freeze", "verify"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.root, args.data_root)
    elif args.action == "verify":
        value = load(args.root, args.freeze_sha256, verify_assets=True)
        result = dict(verified=True, rows=len(value["rows"]), predictions=len(value["predictions"]))
    else:
        rows, targets = planned(args.root, args.data_root)
        forecasts = predictions(args.root, targets)
        result = dict(planned_rows=len(rows), targets=len(targets), resource_cap=CAPS,
                      approvals={m: sum(p["approved"] for p in forecasts if p["method"] == m)
                                 for m in METHODS}, predictions=forecasts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
