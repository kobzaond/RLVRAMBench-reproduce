"""Stage a verified admission-study release; no network writes or deletions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


ADMISSION_SOURCE_FILES = {
    "decision_benchmark.py", "test_decision_benchmark.py",
    "memory_tuner/collect_decision_benchmark.py",
    "memory_tuner/test_collect_decision_benchmark.py",
    "memory_tuner/plan_decision_benchmark.py",
    "memory_tuner/test_artifact_manifest.py",
}
ESTIMATION_SOURCE_FILES = {
    "benchmark.py", "publication-requirements.txt",
    "memory_tuner/estimation_baselines.py",
    "memory_tuner/test_estimation_baselines.py",
    "memory_tuner/model_memory_metadata.py",
    "memory_tuner/test_model_memory_metadata.py",
    "memory_tuner/plan_estimation_study.py",
    "memory_tuner/verify_estimation.py",
    "memory_tuner/test_verify_estimation.py",
    "memory_tuner/collect_estimation_study.py",
    "memory_tuner/test_collect_estimation_study.py",
    "memory_tuner/capture_allocation_accounting.py",
    "memory_tuner/test_capture_allocation_accounting.py",
    "memory_tuner/run_matched_gpu.py",
    "memory_tuner/test_matched_gpu.py",
    "memory_tuner/device_contract.py",
    "memory_tuner/vllm_uuid_compat.py",
    "memory_tuner/test_vllm_uuid_compat.py",
    "memory_tuner/benchmark_v2_env.py",
    "memory_tuner/test_benchmark_v2_env.py",
    "memory_tuner/resolve_hf_snapshot.py",
    "memory_tuner/test_resolve_hf_snapshot.py",
    "memory_tuner/capture_environment.py",
    "memory_tuner/test_capture_environment.py",
    "memory_tuner/gpu_monitor.py",
    "memory_tuner/test_gpu_monitor.py",
    "memory_tuner/code_shape_reward.py",
    "memory_tuner/test_code_shape_reward.py",
    "instrumented_python_packages/sitecustomize.py",
    "instrumented_python_packages/verl/__init__.py",
    "instrumented_python_packages/verl/workers/engine_workers.py",
    "run_matched_gpu.slurm", "run_matched_gpu_preflight.slurm",
    "run_grpo_instrumented.slurm",
}
HOST_CAPACITY_SOURCE_FILES = {
    "memory_tuner/host_capacity_protocol.py", "memory_tuner/test_host_capacity_protocol.py",
    "memory_tuner/collect_host_capacity.py", "memory_tuner/test_collect_host_capacity.py",
    "memory_tuner/run_host_capacity.py", "memory_tuner/test_run_host_capacity.py",
    "run_host_capacity.slurm",
}
HOST_CAPACITY_OUTPUT_FILES = {
    f"{name}.json" for name in
    ("processes", "configurations", "configurations_flat", "pairs", "evaluation",
     "summary", "costs", "provenance", "output_manifest")
} | {f"{name}.csv" for name in ("processes", "configurations", "configurations_flat", "pairs")}
ESTIMATION_COUNT_FIELDS = (
    "planned_processes", "planned_pairs", "planned_configurations",
    "eligible_processes", "completed_processes", "unresolved_processes",
    "known_launched_processes", "unknown_launch_status_processes",
    "resolved_configurations",
)
ESTIMATION_RECONSTRUCTION_FLAGS = (
    "estimation_independent_raw_reconstruction_passed",
    "estimation_reference_outcomes_hidden_during_analysis",
    "estimation_prediction_replay_passed",
    "estimation_prediction_replay_outcomes_hidden",
)
HOST_CAPACITY_RECONSTRUCTION_FLAGS = (
    "host_capacity_independent_raw_reconstruction_passed",
    "host_capacity_reference_outcomes_hidden_during_analysis",
    "host_capacity_prediction_inheritance_replay_passed",
    "host_capacity_prediction_replay_outcomes_hidden",
    "host_capacity_original_prediction_replay_passed",
    "estimation_prediction_replay_host_outcomes_hidden",
)
REPLICATION_PANELS = ("completion_replication", "replication_control")
REPLICATION_WRAPPER_FILES = ("reproduce.py", "replication_reproduction.py")
REPLICATION_ACCOUNTING_CODE_FILES = {
    f"memory_tuner/{name}.py" for name in (
        "__init__", "core", "device_contract", "capture_allocation_accounting",
        "replication_control_accounting")
}
REPLICATION_OUTPUTS = {
    "completion_replication": {"summary.json", "slots.csv", "configurations.csv", "pairs.csv"},
    "replication_control": {
        "replication_control_summary.json", "replication_control_trials.csv",
        "replication_control_blocks.csv", "replication_control_cases.csv"},
}
OFFLOAD_SOURCE_COMMIT = "3d6ecac1852e16a0422c9c72a230187be663f8b5"
OFFLOAD_FREEZE_SHA256 = "5df43f35e0b02c23e4816b7e860c5a613176da41f5adb27e57f79ac2becf1fb9"
OFFLOAD_WRAPPER_FILES = ("reproduce.py", "replication_reproduction.py", "offload_reproduction.py")
OFFLOAD_OUTPUTS = {
    "slots.csv", "configurations.csv", "prediction_scores.csv", "offload_pairs.csv",
    "block_contrasts.csv", "summary.json",
}
OFFLOAD_CONTRACT_FIELDS = {
    "schema_version", "code_sha256", "source_commit", "freeze_sha256", "source_tree_sha256",
    "source_export_receipt_sha256", "source_export_helper_sha256",
    "allocation_accounting_sha256", "output_sha256",
}
OFFLOAD_RECONSTRUCTION_FLAGS = (
    "archived_source_only", "source_and_manifest_verified",
    "independent_raw_reconstruction_passed", "reference_outcomes_hidden",
    "child_namespace_verified", "prediction_replay_passed", "prediction_replay_outcomes_hidden",
    "source_export_receipt_verified", "whole_allocation_accounting_reconstruction_passed",
)


def metadata(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy(source, destination):
    if not source.is_file():
        raise ValueError(f"Required publication input missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_tree_digest(directory):
    if (not directory.is_dir()
            or any(p.is_symlink() for p in (directory, *directory.parents))):
        raise ValueError(f"Missing or linked replication source tree: {directory}")
    files = {}
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Nonregular replication source entry: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = metadata(path)["sha256"]
    if not files:
        raise ValueError("Empty replication source tree")
    return canonical_digest(files)


def validate_offload_contract(contract):
    release_require(
        isinstance(contract, dict) and set(contract) == OFFLOAD_CONTRACT_FIELDS
        and type(contract["schema_version"]) is int and contract["schema_version"] == 1
        and contract["source_commit"] == OFFLOAD_SOURCE_COMMIT
        and contract["freeze_sha256"] == OFFLOAD_FREEZE_SHA256
        and isinstance(contract["code_sha256"], dict)
        and set(contract["code_sha256"]) == set(OFFLOAD_WRAPPER_FILES)
        and isinstance(contract["output_sha256"], dict)
        and set(contract["output_sha256"]) == OFFLOAD_OUTPUTS,
        "Invalid version-bound offload reproduction contract")
    digests = [contract[key] for key in (
        "freeze_sha256", "source_tree_sha256", "source_export_receipt_sha256",
        "source_export_helper_sha256", "allocation_accounting_sha256")]
    digests += list(contract["code_sha256"].values()) + list(contract["output_sha256"].values())
    release_require(all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
                        for value in digests), "Invalid offload SHA256 anchor")


def offload_release_fields(root, code):
    directory = root / "benchmark/offload_factorial"
    source = root / "reproduction_sources/offload_factorial"
    if not any(p.exists() or p.is_symlink() for p in
               (directory, source, root / "output/offload_factorial")):
        return {}
    output = directory / "results"
    release_require(
        output.is_dir() and not output.is_symlink()
        and {p.name for p in output.iterdir()} == OFFLOAD_OUTPUTS
        and all(p.is_file() and not p.is_symlink() for p in output.iterdir()),
        "Complete six-file offload collector output is required")
    tree = source_tree_digest(source / "source")
    release_require({p.name for p in source.iterdir()} == {
        "source", "snapshot_offload_factorial.py", "source-export-report.json"},
        "Unexpected offload source export entries")
    receipt = json.loads((source / "source-export-report.json").read_text())
    helper = metadata(source / "snapshot_offload_factorial.py")["sha256"]
    files = {p.relative_to(source / "source").as_posix(): metadata(p)
             for p in (source / "source").rglob("*") if p.is_file()}
    release_require(
        receipt.get("schema_version") == "offload-source-export-1"
        and receipt.get("status") == "verified_git_free_cpu_export"
        and receipt.get("source_commit") == OFFLOAD_SOURCE_COMMIT
        and receipt.get("freeze_sha256") == OFFLOAD_FREEZE_SHA256
        and receipt.get("files") == files
        and receipt.get("source_file_count") == len(files)
        and receipt.get("source_total_bytes") == sum(item["size_bytes"] for item in files.values())
        and receipt.get("verification_code", {}).get("sha256") == helper,
        "Offload source export receipt differs from its bytes/anchors")
    for name in ("freeze.json", "protocol.json", "matrix.csv", "predictions.json"):
        expected = metadata(source / "source/benchmark/offload_factorial" / name)
        release_require(metadata(directory / name) == expected
                        and (name != "freeze.json" or expected["sha256"] == OFFLOAD_FREEZE_SHA256),
                        "Public offload protocol differs from its frozen source")
    wrappers = {}
    for name in OFFLOAD_WRAPPER_FILES:
        expected = metadata(code / name)["sha256"]
        release_require(metadata(root / "reproduction_sources/driver" / name)["sha256"] == expected,
                        "Archived offload driver differs from public code")
        wrappers[name] = expected
    contract = dict(
        schema_version=1, code_sha256=wrappers, source_commit=OFFLOAD_SOURCE_COMMIT,
        freeze_sha256=OFFLOAD_FREEZE_SHA256, source_tree_sha256=tree,
        source_export_receipt_sha256=metadata(source / "source-export-report.json")["sha256"],
        source_export_helper_sha256=helper,
        allocation_accounting_sha256=metadata(directory / "allocation_accounting.json")["sha256"],
        output_sha256={name: metadata(output / name)["sha256"] for name in sorted(OFFLOAD_OUTPUTS)},
    )
    validate_offload_contract(contract)
    return {"offload_reproduction": contract}


def verify_offload_reconstruction(stage, release, result, manifest):
    contract = release.get("offload_reproduction")
    report = result.get("offload_reproduction")
    present = (stage / "benchmark/offload_factorial").exists()
    if contract is None and report is None and not present:
        return
    validate_offload_contract(contract)
    release_require(present and isinstance(report, dict)
                    and result.get("offload_reproduction_contract_sha256") == canonical_digest(contract),
                    "Offload release and reconstruction contracts disagree")
    release_require(
        type(report.get("panel_files_matched")) is int and report["panel_files_matched"] == 6
        and all(report.get(flag) is True for flag in OFFLOAD_RECONSTRUCTION_FLAGS)
        and report.get("fresh_refit_performed") is False
        and type(report.get("prediction_rows_replayed")) is int
        and report["prediction_rows_replayed"] == 288,
        "Full isolated offload reconstruction and unchanged prediction replay have not passed")
    for key in ("source_commit", "freeze_sha256", "source_tree_sha256",
                "source_export_receipt_sha256", "source_export_helper_sha256",
                "allocation_accounting_sha256", "output_sha256"):
        release_require(report.get(key) == contract[key], f"Offload reconstructed anchor differs: {key}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = {row["path"]: row for row in rows}
    release_require(len(entries) == len(rows), "Duplicate offload manifest entries")
    for name, expected in contract["code_sha256"].items():
        manifest_anchor(entries, "reproduction_sources/driver/" + name, expected)
    base = "reproduction_sources/offload_factorial/"
    manifest_anchor(entries, base + "source-export-report.json", contract["source_export_receipt_sha256"])
    manifest_anchor(entries, base + "snapshot_offload_factorial.py", contract["source_export_helper_sha256"])
    prefix = base + "source/"
    tree = {name[len(prefix):]: row["sha256"] for name, row in entries.items()
            if name.startswith(prefix)}
    release_require(tree and canonical_digest(tree) == contract["source_tree_sha256"],
                    "Offload source tree differs from archive")
    design = "benchmark/offload_factorial/"
    for name in ("freeze.json", "protocol.json", "matrix.csv", "predictions.json"):
        expected = tree.get(design + name)
        manifest_anchor(entries, design + name, expected)
        release_require(metadata(stage / design / name)["sha256"] == expected
                        and (name != "freeze.json" or expected == contract["freeze_sha256"]),
                        "Staged offload protocol differs from its archived source")
    manifest_anchor(entries, design + "allocation_accounting.json", contract["allocation_accounting_sha256"])
    release_require(metadata(stage / design / "allocation_accounting.json")["sha256"]
                    == contract["allocation_accounting_sha256"], "Staged offload accounting differs")
    output = stage / design / "results"
    release_require({p.name for p in output.iterdir()} == OFFLOAD_OUTPUTS,
                    "Staged offload output inventory differs")
    for name, expected in contract["output_sha256"].items():
        manifest_anchor(entries, design + "results/" + name, expected)
        release_require(metadata(output / name)["sha256"] == expected,
                        "Staged offload scientific output differs")
    summary = json.loads((output / "summary.json").read_text())
    release_require(report.get("whole_allocation_accounting") == summary.get("allocation_accounting")
                    and isinstance(summary.get("allocation_accounting"), dict)
                    and type(summary["allocation_accounting"].get("allocations")) is int,
                    "Offload whole-allocation costs differ from reconstructed summary")
    locations = report.get("relocated_provenance")
    release_require(isinstance(locations, dict) and set(locations) == {"raw_root", "source_snapshot"},
                    "Offload relocation provenance is missing")
    for name, value in locations.items():
        release_require(isinstance(value, dict) and set(value) == {"published", "reconstructed"}
                        and value["published"] == summary.get(name)
                        and isinstance(value["reconstructed"], str)
                        and Path(value["reconstructed"]).is_absolute()
                        and ".." not in Path(value["reconstructed"]).parts,
                        "Offload relocation provenance differs")


def replication_release_fields(root, code):
    panels = {}
    for name in REPLICATION_PANELS:
        directory = root / "benchmark" / name
        source = root / "reproduction_sources" / name
        if not any(p.exists() or p.is_symlink() for p in
                   (directory, source, root / "output" / name)):
            continue
        output = directory / "results"
        if (not output.is_dir() or output.is_symlink()
                or {p.name for p in output.iterdir()} != REPLICATION_OUTPUTS[name]
                or any(not p.is_file() or p.is_symlink() for p in output.iterdir())):
            raise ValueError(f"Complete replication collector output is required: {name}")
        if name == "completion_replication":
            amendment_name = "completion-runtime-amendment-01.json"
            freeze = source / "amended/benchmark/completion_replication/freeze.json"
            panel = {
                "source_tree_sha256": {
                    cohort: source_tree_digest(source / cohort)
                    for cohort in ("original", "amended")},
                "freeze_sha256": metadata(freeze)["sha256"],
                "runtime_amendment_sha256": metadata(source / amendment_name)["sha256"],
                "amendment_filename": amendment_name,
                "allocation_accounting_sha256": metadata(
                    directory / "allocation_accounting.json")["sha256"],
            }
        else:
            design = source / "benchmark/replication_control"
            panel = {
                "source_tree_sha256": {"amended": source_tree_digest(source)},
                "execution_seal_sha256": metadata(
                    design / "replication_control_execution_seal.json")["sha256"],
                "runtime_amendment_sha256": metadata(
                    design / "replication_control_runtime_amendment.json")["sha256"],
            }
            accounting = directory / "allocation_accounting.json"
            if accounting.exists():
                panel.update(
                    allocation_accounting_sha256=metadata(accounting)["sha256"],
                    accounting_code_sha256={
                        name: metadata(root / name)["sha256"]
                        for name in sorted(REPLICATION_ACCOUNTING_CODE_FILES)},
                )
        panels[name] = panel
    if not panels:
        return {}
    wrapper_hashes = {name: metadata(code / name)["sha256"]
                      for name in REPLICATION_WRAPPER_FILES}
    for name, expected in wrapper_hashes.items():
        if metadata(root / "reproduction_sources/driver" / name)["sha256"] != expected:
            raise ValueError("Archived reproduction driver differs from the public code")
    return {"replication_reproduction": {
        "schema_version": 1,
        "code_sha256": wrapper_hashes,
        "panels": panels,
    }}


def release_require(condition, message):
    if not condition:
        raise ValueError(message)


def manifest_anchor(entries, name, expected):
    release_require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected)
                    and name in entries and entries[name]["sha256"] == expected,
                    f"Replication digest differs from the archive: {name}")


def verify_replication_reconstruction(stage, release, result, manifest):
    contract = release.get("replication_reproduction")
    present = {name for name in REPLICATION_PANELS
               if (stage / "benchmark" / name).exists()}
    actual = result.get("replication_reproduction", {})
    if contract is None and not present and not actual:
        return
    if (not isinstance(contract, dict)
            or set(contract) != {"schema_version", "code_sha256", "panels"}
            or type(contract.get("schema_version")) is not int or contract["schema_version"] != 1
            or not isinstance(contract["panels"], dict)
            or not isinstance(contract["code_sha256"], dict)
            or set(contract["code_sha256"]) != set(REPLICATION_WRAPPER_FILES)
            or set(contract.get("panels", {})) != present
            or set(actual) != present or not present):
        raise ValueError("Replication release/reconstruction panels disagree")
    release_require(result.get("replication_reproduction_contract_sha256") == canonical_digest(contract),
                    "Replication contract differs from the verified reconstruction")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = {row["path"]: row for row in rows}
    release_require(len(entries) == len(rows), "Duplicate replication manifest entries")
    for name, expected in contract["code_sha256"].items():
        manifest_anchor(entries, "reproduction_sources/driver/" + name, expected)
    required_flags = (
        "archived_source_only", "source_and_manifest_verified",
        "independent_raw_reconstruction_passed", "reference_outcomes_hidden",
        "child_namespace_verified",
    )
    for name, settings in contract["panels"].items():
        common = {"source_tree_sha256", "runtime_amendment_sha256"}
        required = (common | {"freeze_sha256", "amendment_filename", "allocation_accounting_sha256"}
                    if name == "completion_replication" else common | {"execution_seal_sha256"})
        cost_fields = {"allocation_accounting_sha256", "accounting_code_sha256"}
        release_require(isinstance(settings, dict) and (
            set(settings) == required
            or name == "replication_control" and set(settings) == required | cost_fields),
            f"Invalid replication panel contract: {name}")
        cohorts = {"original", "amended"} if name == "completion_replication" else {"amended"}
        release_require(isinstance(settings["source_tree_sha256"], dict)
                        and set(settings["source_tree_sha256"]) == cohorts,
                        f"Incomplete replication source cohorts: {name}")
        report = actual[name]
        if (report.get("panel_files_matched") != 4
                or any(report.get(flag) is not True for flag in required_flags)
                or report.get("runtime_amendment_sha256") != settings["runtime_amendment_sha256"]
                or report.get("source_tree_sha256") != settings["source_tree_sha256"]):
            raise ValueError(f"Isolated replication reconstruction has not passed: {name}")
        for cohort, expected in settings["source_tree_sha256"].items():
            release_require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected),
                            "Invalid replication source-tree digest")
            prefix = f"reproduction_sources/{name}/"
            if name == "completion_replication":
                prefix += cohort + "/"
            inventory = {path[len(prefix):]: row["sha256"] for path, row in entries.items()
                         if path.startswith(prefix)}
            if not inventory or canonical_digest(inventory) != expected:
                raise ValueError(f"Replication source anchor differs from the archive: {name}")
        if name == "completion_replication":
            release_require(settings["amendment_filename"] == "completion-runtime-amendment-01.json",
                            "Unexpected completion amendment filename")
            base = "reproduction_sources/completion_replication/"
            manifest_anchor(entries, base + settings["amendment_filename"],
                            settings["runtime_amendment_sha256"])
            for cohort in ("original", "amended"):
                manifest_anchor(entries, base + cohort + "/benchmark/completion_replication/freeze.json",
                                settings["freeze_sha256"])
            release_require(report.get("source_export_receipt_verified") is True,
                            "Completion source-export receipt was not verified")
            manifest_anchor(entries, base + "source-export-report.json",
                            report.get("source_export_receipt_sha256"))
            timing = report.get("collection_cpu_wall_seconds", {})
            if (report.get("prediction_replay_passed") is not True
                    or report.get("prediction_replay_outcomes_hidden") is not True
                    or report.get("fresh_refit_performed") is not False
                    or report.get("allocation_accounting_sha256")
                    != settings["allocation_accounting_sha256"]
                    or not isinstance(timing, dict)
                    or set(timing) != {"published_seconds", "reconstructed_seconds"}
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v <= 0 for v in timing.values())):
                raise ValueError("Completion prediction replay or timing disclosure has not passed")
            if metadata(stage / "benchmark/completion_replication/allocation_accounting.json")[
                    "sha256"] != settings["allocation_accounting_sha256"]:
                raise ValueError("Completion accounting differs from the staged input")
        else:
            base = "reproduction_sources/replication_control/"
            design = "benchmark/replication_control/"
            manifest_anchor(entries, base + design + "replication_control_runtime_amendment.json",
                            settings["runtime_amendment_sha256"])
            for cohort in ("", "replication_control_base_source/"):
                manifest_anchor(entries, base + cohort + design + "replication_control_execution_seal.json",
                                settings["execution_seal_sha256"])
            accounting = stage / "benchmark/replication_control/allocation_accounting.json"
            release_require(accounting.exists() == ("allocation_accounting_sha256" in settings),
                            "Control accounting presence differs from its contract")
            if accounting.exists():
                hashes = settings["accounting_code_sha256"]
                release_require(isinstance(hashes, dict) and set(hashes) == REPLICATION_ACCOUNTING_CODE_FILES,
                                "Control accounting code contract is incomplete")
                for path, expected in hashes.items():
                    manifest_anchor(entries, path, expected)
                release_require(report.get("whole_allocation_accounting_reconstruction_passed") is True
                                and report.get("allocation_accounting_sha256")
                                == settings["allocation_accounting_sha256"]
                                and report.get("accounting_code_sha256") == hashes,
                                "Control allocation accounting reconstruction has not passed")
                release_require(metadata(accounting)["sha256"] == settings["allocation_accounting_sha256"],
                                "Control accounting differs from the staged input")
                from memory_tuner.capture_allocation_accounting import validate_record
                records = json.loads(accounting.read_text())
                expected_allocations = []
                for item in records:
                    value = validate_record(item)
                    release_require(value["allocated_gpu_count"] == 2,
                                    "Control accounting GPU count differs")
                    start, end = value["started_at_epoch"], value["ended_at_epoch"]
                    first, last = value["batch_started_at_epoch"], value["batch_ended_at_epoch"]
                    expected_allocations.append({
                        "block_id": item["pair_id"], "job_id": value["job_id"],
                        "scheduler_state": value["scheduler_state"],
                        "whole_allocation_gpu_seconds": 2 * (end - start),
                        "batch_step_gpu_seconds": 2 * (last - first) if last is not None else None,
                    })
                cost = report.get("whole_allocation_accounting", {})
                expected_allocations.sort(key=lambda r: int(r["job_id"]))
                release_require(
                    0 < len(records) <= 36 and len({r["job_id"] for r in expected_allocations}) == len(records)
                    and cost.get("allocation_count") == len(records)
                    and cost.get("allocations") == expected_allocations
                    and cost.get("whole_allocation_gpu_seconds") ==
                    sum(r["whole_allocation_gpu_seconds"] for r in expected_allocations),
                    "Reconstructed control costs differ from the bound scheduler records")
        if "allocation_accounting_sha256" in settings:
            manifest_anchor(entries, f"benchmark/{name}/allocation_accounting.json",
                            settings["allocation_accounting_sha256"])


def verify_archives(built, directory=None):
    for entry in built["archives"]:
        path = (directory / Path(entry["path"]).name
                if directory is not None else Path(entry["path"]))
        if metadata(path) != {
                key: entry[key] for key in ("sha256", "size_bytes")}:
            raise ValueError(f"Archive differs from frozen release metadata: {path.name}")


def archived_name(name):
    """Map a loose public file to its evidence-manifest path."""
    if name == "README.md":
        return "paper/huggingface_dataset_card.md"
    if name == "LICENSE":
        return "LICENSE-RLVRAMBench"
    if name in ("BENCHMARK.md", "DATA_DICTIONARY.md", "RIGHTS.md"):
        return name
    if name.startswith("benchmark/") or name.startswith("paper/figures/"):
        return name
    if name.startswith("tables/review_revision/"):
        return name.replace("tables/review_revision/", "profiles/review_revision/", 1)
    if name.startswith("tables/"):
        return name.replace("tables/", "profiles/standard_grpo/", 1)
    if name == "paper/RLVRAMBench_IEEE_Access_source.zip":
        return "paper/submission/RLVRAMBench_IEEE_Access_source.zip"
    if name.startswith("paper/"):
        return name.replace("paper/", "paper/submission/ieee_access_submission/", 1)
    if name == "verification/admission-source-package.json":
        return "paper/submission/ieee_access_submission/export_verification.json"
    return None


def verify_loose_payload(stage, manifest):
    with manifest.open(newline="") as handle:
        entries = {row["path"]: row for row in csv.DictReader(handle)}
    # Derive completeness from the authenticated archive, not from whichever
    # files happen to remain in staging.
    expected_public = {}
    fixed = {
        "paper/huggingface_dataset_card.md": "README.md",
        "LICENSE-RLVRAMBench": "LICENSE",
        "paper/submission/RLVRAMBench_IEEE_Access_source.zip":
            "paper/RLVRAMBench_IEEE_Access_source.zip",
        "paper/submission/ieee_access_submission/export_verification.json":
            "verification/admission-source-package.json",
    }
    for stem in ("RLVRAMBench_IEEE_Access_submission", "RLVRAMBench_supplement"):
        for suffix in (".tex", ".pdf"):
            fixed[f"paper/submission/ieee_access_submission/{stem}{suffix}"] = (
                f"paper/{stem}{suffix}")
    figure_names = {
        f"paper/figures/{stem}{suffix}"
        for stem in ("standard_grpo_boundary", "standard_grpo_mechanisms",
                     "standard_grpo_temporal", "review_instrumentation_batch")
        for suffix in (".pdf", ".png")
    }
    for original in entries:
        public = fixed.get(original)
        if (original in ("BENCHMARK.md", "DATA_DICTIONARY.md", "RIGHTS.md")
                or original.startswith("benchmark/") or original in figure_names):
            public = original
        elif original.startswith("profiles/standard_grpo/"):
            public = original.replace("profiles/standard_grpo/", "tables/", 1)
        elif original.startswith("profiles/review_revision/"):
            public = original.replace("profiles/review_revision/", "tables/review_revision/", 1)
        if public is not None:
            expected_public[public] = original
    for name in expected_public:
        if not (stage / name).is_file():
            raise ValueError(f"Required loose publication file is missing: {name}")
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        name = str(path.relative_to(stage))
        original = archived_name(name)
        if original is None:
            if (name in ("release.json", "upload-manifest.json",
                         "verification/admission-reproduction.json")
                    or name.startswith("releases/")):
                continue
            raise ValueError(f"Unmapped publication payload: {name}")
        expected = entries.get(original)
        if expected is None or metadata(path) != {
                "sha256": expected["sha256"],
                "size_bytes": int(expected["size_bytes"])}:
            raise ValueError(f"Loose publication file differs from frozen archive: {name}")


def collector_release_inventory(root, cohort="estimation"):
    """Read complete collector outputs; unresolved slots are valid evidence.

    This validates the published inventory, not the raw experiment again.
    Independent reconstruction and prediction replay are separate final gates.
    """
    directory = root / "benchmark" / cohort
    output = directory / "results"
    if not (output / "summary.json").exists():
        return None
    summary = json.loads((output / "summary.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    inventory = json.loads((output / "output_manifest.json").read_text())
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Empty estimation collector output manifest")
    for name, expected in inventory.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or name == "output_manifest.json"
                or relative.suffix not in (".json", ".csv")):
            raise ValueError(f"Unsafe estimation derived-file name: {name}")
        path = output / relative
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(output.resolve())
                or metadata(path) != expected):
            raise ValueError(f"Estimation collector output differs: {name}")
    actual_files = {
        str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
    }
    if actual_files != set(inventory) | {"output_manifest.json"}:
        raise ValueError("Estimation collector output file set differs")
    counts = {name: summary[name] for name in ESTIMATION_COUNT_FIELDS}
    counts["unresolved_configurations"] = evaluation["unresolved_configurations"]
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Estimation counts must be nonnegative integers")
    if (counts["eligible_processes"] + counts["unresolved_processes"]
            != counts["planned_processes"]
            or counts["completed_processes"] > counts["eligible_processes"]
            or counts["known_launched_processes"] + counts["unknown_launch_status_processes"]
            > counts["planned_processes"]
            or counts["resolved_configurations"] + counts["unresolved_configurations"]
            != counts["planned_configurations"]
            or evaluation["resolved_configurations"] != counts["resolved_configurations"]
            or evaluation["planned_configurations"] != counts["planned_configurations"]):
        raise ValueError("Estimation collector counts disagree")
    return directory, counts, provenance, actual_files


def estimation_release_fields(root):
    inventory = collector_release_inventory(root)
    if inventory is None:
        return {}
    directory, counts, provenance, actual_files = inventory
    protocol = json.loads((directory / "protocol.json").read_text())
    seal = json.loads((directory / "prediction_freeze.json").read_text())
    protocol_hash = metadata(directory / "protocol.json")["sha256"]
    seal_hash = metadata(directory / "prediction_freeze.json")["sha256"]
    if (provenance["protocol_sha256"] != protocol_hash
            or seal["protocol_sha256"] != protocol_hash
            or provenance["prediction_seal_sha256"] != seal_hash):
        raise ValueError("Estimation collector is bound to a different frozen seal")
    with (directory / "matrix.csv").open(newline="") as handle:
        matrix = list(csv.DictReader(handle))
    pair_ids = {row["pair_id"] for row in matrix}
    if (len(matrix) != counts["planned_processes"]
            or len(pair_ids) != counts["planned_pairs"]
            or protocol["planned_processes"] != counts["planned_processes"]
            or protocol["planned_pairs"] != counts["planned_pairs"]
            or protocol["target_configurations"] != counts["planned_configurations"]):
        raise ValueError("Estimation planned coverage differs from the frozen protocol")
    commits = provenance["execution_commits_expected"]
    if (not isinstance(commits, dict) or set(commits) != pair_ids
            or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value)
                   for value in commits.values())):
        raise ValueError("Estimation execution commits must cover every planned pair")
    commits_path = directory / "execution_commits.json"
    if (json.loads(commits_path.read_text()) != commits
            or metadata(commits_path)["sha256"]
            != provenance["execution_commits_file_sha256"]):
        raise ValueError("Estimation execution-map bytes differ from collector input")
    accounting_path = directory / "allocation_accounting.json"
    accounting_hash = provenance.get("allocation_accounting_sha256")
    if accounting_path.exists() or accounting_hash is not None:
        if not accounting_path.is_file() or metadata(accounting_path)["sha256"] != accounting_hash:
            raise ValueError("Estimation accounting bytes differ from collector input")
    return {
        "estimation_protocol_version": protocol["protocol_version"],
        "expected_estimation_derived_files": len(actual_files),
        "estimation_counts": counts,
        "estimation_protocol_sha256": protocol_hash,
        "estimation_prediction_seal_sha256": seal_hash,
        "estimation_execution_commits": commits,
        "estimation_execution_commits_sha256": metadata(commits_path)["sha256"],
        "estimation_allocation_accounting_sha256": accounting_hash,
    }


def host_capacity_release_fields(root):
    """A separate cohort, anchored to the original predictor, never pooled."""
    inventory = collector_release_inventory(root, "host_capacity")
    if inventory is None:
        return {}
    directory, counts, provenance, actual_files = inventory
    if actual_files != HOST_CAPACITY_OUTPUT_FILES:
        raise ValueError("Host-capacity collector must provide exactly its 13 output files")
    original = estimation_release_fields(root)
    if not original:
        raise ValueError("Host-capacity requires the separately replayed original estimation cohort")
    protocol = json.loads((directory / "protocol.json").read_text())
    freeze = json.loads((directory / "study_freeze.json").read_text())
    protocol_hash = metadata(directory / "protocol.json")["sha256"]
    freeze_hash = metadata(directory / "study_freeze.json")["sha256"]
    original_protocol = original["estimation_protocol_sha256"]
    original_seal = original["estimation_prediction_seal_sha256"]
    if (protocol["protocol_version"] != "host-capacity-1.0"
            or protocol["group"] != "estimation_host_capacity"
            or provenance["protocol_sha256"] != protocol_hash
            or freeze["protocol_sha256"] != protocol_hash
            or provenance["study_freeze_sha256"] != freeze_hash
            or provenance["study_frozen_at_utc"] != freeze["frozen_at_utc"]
            or protocol["original_protocol_sha256"] != original_protocol
            or provenance["original_protocol_sha256"] != original_protocol
            or any(value != original_seal for value in (
                protocol["original_prediction_seal_sha256"],
                freeze["original_prediction_seal_sha256"],
                provenance["original_prediction_seal_sha256"],
                provenance["prediction_seal_sha256"]))):
        raise ValueError("Host-capacity collector differs from its study freeze/original prediction seal")
    if set(freeze["files_sha256"]) != {"matrix.csv", "targets.csv", "predictions.json"}:
        raise ValueError("Host-capacity freeze does not bind all inherited inputs")
    for name, expected in freeze["files_sha256"].items():
        path = directory / name
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())
                or metadata(path)["sha256"] != expected):
            raise ValueError(f"Host-capacity frozen input differs: {name}")
    with (directory / "matrix.csv").open(newline="") as handle:
        matrix = list(csv.DictReader(handle))
    with (directory / "targets.csv").open(newline="") as handle:
        targets = list(csv.DictReader(handle))
    pairs = {row["pair_id"] for row in matrix}
    configurations = {row["configuration_id"] for row in matrix}
    if (len(matrix) != counts["planned_processes"] or len(pairs) != counts["planned_pairs"]
            or len(configurations) != counts["planned_configurations"]
            or configurations != {row["configuration_id"] for row in targets}
            or len(targets) != len(configurations)
            or protocol["planned_processes"] != len(matrix)
            or protocol["planned_pairs"] != len(pairs)
            or protocol["target_configurations"] != len(configurations)):
        raise ValueError("Host-capacity planned counts differ from frozen inputs")
    commits = provenance["execution_commits_expected"]
    commits_path, accounting_path = (directory / name for name in
                                      ("execution_commits.json", "allocation_accounting.json"))
    if (not isinstance(commits, dict) or set(commits) != pairs
            or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value)
                   for value in commits.values())
            or json.loads(commits_path.read_text()) != commits
            or metadata(commits_path)["sha256"] != provenance["execution_commits_file_sha256"]):
        raise ValueError("Host-capacity execution map differs from the complete frozen pair inventory")
    accounting_hash = provenance.get("allocation_accounting_sha256")
    if (not accounting_path.is_file() or accounting_path.is_symlink()
            or metadata(accounting_path)["sha256"] != accounting_hash):
        raise ValueError("Host-capacity accounting input must be explicitly supplied and hash-bound")
    return {
        "expected_host_capacity_derived_files": len(actual_files),
        "host_capacity_protocol_version": protocol["protocol_version"],
        "host_capacity_counts": counts,
        "host_capacity_protocol_sha256": protocol_hash,
        "host_capacity_study_freeze_sha256": freeze_hash,
        "host_capacity_original_protocol_sha256": original_protocol,
        "host_capacity_prediction_seal_sha256": original_seal,
        "host_capacity_execution_commits": commits,
        "host_capacity_execution_commits_sha256": metadata(commits_path)["sha256"],
        "host_capacity_allocation_accounting_sha256": accounting_hash,
    }


def verify_estimation_reconstruction(stage, release, result):
    expected = estimation_release_fields(stage)
    enabled = any(key.startswith("estimation_") or key == "expected_estimation_derived_files"
                  for key in release)
    if not expected and not enabled:
        return
    if not expected:
        raise ValueError("Estimation reconstruction lacks published collector outputs")
    for key, value in expected.items():
        if json.dumps(release.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Estimation release metadata differs from collector evidence: {key}")
    for result_key, release_key in (
        ("estimation_panel_files_matched", "expected_estimation_derived_files"),
        ("estimation_counts", "estimation_counts"),
        ("estimation_prediction_seal_sha256", "estimation_prediction_seal_sha256"),
        ("estimation_execution_commits", "estimation_execution_commits"),
    ):
        if (json.dumps(result.get(result_key), sort_keys=True)
                != json.dumps(expected[release_key], sort_keys=True)):
            raise ValueError(f"Estimation reconstruction differs: {result_key}")
    if any(result.get(key) is not True for key in ESTIMATION_RECONSTRUCTION_FLAGS):
        raise ValueError("Isolated estimation reconstruction and prediction replay have not passed")
    if type(result.get("estimation_fresh_refit_compared", False)) is not bool:
        raise ValueError("Estimation fresh-refit status must be reported separately as a boolean")


def verify_host_capacity_reconstruction(stage, release, result):
    expected = host_capacity_release_fields(stage)
    enabled = any(key.startswith("host_capacity_") or key == "expected_host_capacity_derived_files"
                  for key in release)
    if not expected and not enabled:
        return
    if not expected:
        raise ValueError("Host-capacity reconstruction lacks published collector outputs")
    for key, value in expected.items():
        if json.dumps(release.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Host-capacity release metadata differs from collector evidence: {key}")
        result_key = "host_capacity_panel_files_matched" if key == "expected_host_capacity_derived_files" else key
        if json.dumps(result.get(result_key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"Host-capacity reconstruction differs: {result_key}")
    if any(result.get(key) is not True for key in HOST_CAPACITY_RECONSTRUCTION_FLAGS):
        raise ValueError("Isolated host-capacity reconstruction and frozen inheritance replay have not passed")
    if result.get("host_capacity_fresh_refit_performed") is not False:
        raise ValueError("Host-capacity must inherit predictions without fitting")


def prepare(root, code, stage, release_metadata, commit):
    if stage.exists():
        raise FileExistsError(f"Use a new staging directory: {stage}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if len(commit) != 40 or commit != actual_commit:
        raise ValueError("A full committed source identifier is required")
    built = json.loads(release_metadata.read_text())
    old = json.loads((code / "release.json").read_text())
    inventory = json.loads(
        (root / "benchmark/decision/results/collection_summary.json").read_text())
    if inventory["planned_slots"] != 48:
        raise ValueError("Wrong frozen admission panel")
    archive = Path(built["archives"][0]["path"])
    verify_archives(built)
    estimation = estimation_release_fields(root)
    host_capacity = host_capacity_release_fields(root)
    replication = replication_release_fields(root, code)
    offload = offload_release_fields(root, code)
    provenance = json.loads((code / "source-provenance.json").read_text())
    names = set(provenance["files"]) | ADMISSION_SOURCE_FILES
    if estimation:
        names.update(ESTIMATION_SOURCE_FILES)
    if host_capacity:
        names.update(HOST_CAPACITY_SOURCE_FILES)
    if (replication.get("replication_reproduction", {}).get("panels", {})
            .get("replication_control", {}).get("accounting_code_sha256")):
        names.update(REPLICATION_ACCOUNTING_CODE_FILES)
        names.add("memory_tuner/test_replication_control_accounting.py")
    for name in names:
        if not (root / name).is_file():
            raise ValueError(f"Required publication input missing: {root / name}")
    stage.mkdir(parents=True)
    release = {
        **old, "release": built["tag"], "source_commit": commit,
        "archive": {"path": "releases/" + archive.name, **metadata(archive)},
        "manifest_entries": built["manifest_entries"],
        "archive_members": built["review_members"],
        "prior_immutable_release_preserved": old["release"],
        "benchmark_evaluator_version": "1.1",
        "decision_protocol_version": "1.0",
        "decision_cases": 4, "decision_candidates": 12,
        "decision_planned_slots": inventory["planned_slots"],
        "decision_recorded_attempts": inventory["recorded_attempts"],
        "decision_validated_memory_outcomes": inventory["validated_memory_outcomes"],
        "expected_decision_derived_files": len(list(
            (root / "benchmark/decision/results").iterdir())),
        "expected_curated_benchmark_files": len([
            p for p in (root / "benchmark").iterdir() if p.is_file()]),
        "visibility": "public",
    }
    # A prior release may have different optional evidence. Never inherit its
    # counts or enable its gates without the current collector's output.
    for key in list(release):
        if (key.startswith(("estimation_", "host_capacity_", "offload_"))
                or key in {"expected_estimation_derived_files", "expected_host_capacity_derived_files",
                           "replication_reproduction"}):
            del release[key]
    release.update(estimation)
    release.update(host_capacity)
    release.update(replication)
    release.update(offload)
    release.pop("huggingface_revision", None)
    save(stage / "release.json", release)
    save(code / "release.json", {**release, "huggingface_revision": None})
    copy(archive, stage / release["archive"]["path"])
    copy(Path(built["archives"][1]["path"]), stage / "releases" /
         Path(built["archives"][1]["path"]).name)
    copy(release_metadata, stage / "releases" / release_metadata.name)
    copy(root / "paper/huggingface_dataset_card.md", stage / "README.md")
    copy(root / "LICENSE-RLVRAMBench", stage / "LICENSE")
    for name in ("BENCHMARK.md", "DATA_DICTIONARY.md", "RIGHTS.md"):
        copy(root / name, stage / name)
    for path in (root / "benchmark").rglob("*"):
        if path.is_file():
            relative = path.relative_to(root)
            copy(path, stage / relative)
            copy(path, code / relative)
    for relative, destination in (
            ("profiles/standard_grpo", "tables"),
            ("profiles/review_revision", "tables/review_revision")):
        for path in (root / relative).iterdir():
            if path.suffix in (".csv", ".json"):
                copy(path, stage / destination / path.name)
    submission = root / "paper/submission/ieee_access_submission"
    for stem in ("RLVRAMBench_IEEE_Access_submission", "RLVRAMBench_supplement"):
        for extension in (".tex", ".pdf"):
            copy(submission / (stem + extension), stage / "paper" / (stem + extension))
    copy(root / "paper/submission/RLVRAMBench_IEEE_Access_source.zip",
         stage / "paper/RLVRAMBench_IEEE_Access_source.zip")
    for stem in ("standard_grpo_boundary", "standard_grpo_mechanisms",
                 "standard_grpo_temporal", "review_instrumentation_batch"):
        for extension in (".pdf", ".png"):
            copy(root / "paper/figures" / (stem + extension),
                 stage / "paper/figures" / (stem + extension))
    copy(submission / "export_verification.json",
         stage / "verification/admission-source-package.json")
    entries = {}
    for name in sorted(names):
        copy(root / name, code / name)
        entries[name] = metadata(code / name)
    provenance.update({
        "source_commit": commit, "files": entries,
        "selection": "CPU reconstruction closure, static transfer suite and independently evaluated admission panel"
                     + ("; sealed estimation replay and independent paired-study reconstruction"
                        if estimation else "")
                     + ("; separately frozen host-capacity inheritance and raw reconstruction"
                        if host_capacity else "")
                     + ("; version-bound archived sources for the two separate replication panels"
                        if replication else "")
                     + ("; unchanged offload factorial source, prediction replay and six-file reconstruction"
                        if offload else ""),
    })
    save(code / "source-provenance.json", provenance)
    print(json.dumps({"release": built["tag"], "source_files": len(entries),
                      "stage": str(stage)}, indent=2))


def finalize(stage, reproduction):
    result = json.loads(reproduction.read_text())
    release = json.loads((stage / "release.json").read_text())
    built = json.loads((stage / "releases" /
                        f"RLVRAMBench-release-{release['release']}.json").read_text())
    verify_archives(built, stage / "releases")
    if metadata(stage / release["archive"]["path"]) != {
            key: release["archive"][key] for key in ("sha256", "size_bytes")}:
        raise ValueError("Staged archive differs from release.json")
    if not (result["status"] == "passed"
            and result["archive_sha256"] == release["archive"]["sha256"]
            and result["archive_members"] == release["archive_members"] == built["review_members"]
            and result["manifest_entries_verified"] == release["manifest_entries"] == built["manifest_entries"]
            and result["derived_csv_json_files_matched"] == release["expected_derived_files"]
            and result["revised_csv_json_files_matched"] == release["expected_review_derived_files"]
            and result["curated_benchmark_files_matched"] == release["expected_curated_benchmark_files"]
            and result["decision_panel_files_matched"] == release["expected_decision_derived_files"]
            and result["decision_validated_memory_outcomes"] == release["decision_validated_memory_outcomes"]
            and result["original_project_and_reference_tables_hidden"]
            and result["decision_reference_outcomes_hidden_during_analysis"]
            and result["curated_references_hidden_during_analysis"]
            and result["child_namespace_reference_inaccessibility_checked"]):
        raise ValueError("Full isolated reconstruction has not passed")
    manifest = (reproduction.parent / "reference-profiles/benchmark/"
                "publication-artifact-manifest.csv")
    if metadata(manifest)["sha256"] != result["evidence_manifest_sha256"]:
        raise ValueError("Extracted manifest changed since reconstruction")
    verify_loose_payload(stage, manifest)
    verify_estimation_reconstruction(stage, release, result)
    verify_host_capacity_reconstruction(stage, release, result)
    verify_replication_reconstruction(stage, release, result, manifest)
    verify_offload_reconstruction(stage, release, result, manifest)
    copy(reproduction, stage / "verification/admission-reproduction.json")
    files = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "upload-manifest.json":
            if path.is_symlink():
                raise ValueError(f"Unexpected publication symlink: {path}")
            files[str(path.relative_to(stage))] = metadata(path)
    save(stage / "upload-manifest.json", {"algorithm": "sha256", "files": files})
    print(json.dumps({"public_payload_files": len(files) + 1}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--release-metadata", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--reproduction", type=Path)
    args = parser.parse_args()
    if args.reproduction:
        finalize(args.stage, args.reproduction)
    else:
        if not all((args.root, args.code, args.release_metadata, args.source_commit)):
            parser.error("Preparation requires source root, code root, release metadata and commit")
        prepare(args.root.resolve(), args.code.resolve(), args.stage.resolve(),
                args.release_metadata.resolve(), args.source_commit)


if __name__ == "__main__":
    main()
