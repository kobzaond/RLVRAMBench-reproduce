"""Release-integrity regressions using synthetic files, never training data."""
import csv
from copy import deepcopy
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest

from memory_tuner import stage_decision_release as staging
from memory_tuner.artifact_manifest import collect_files


PUBLIC_PATHS = {
    "README.md": "paper/huggingface_dataset_card.md",
    "LICENSE": "LICENSE-RLVRAMBench",
    "BENCHMARK.md": "BENCHMARK.md",
    "DATA_DICTIONARY.md": "DATA_DICTIONARY.md",
    "RIGHTS.md": "RIGHTS.md",
    "benchmark/outcomes.csv": "benchmark/outcomes.csv",
    "benchmark/decision/protocol.json": "benchmark/decision/protocol.json",
    "benchmark/decision/results/candidate_results.csv":
        "benchmark/decision/results/candidate_results.csv",
    "tables/summary.json": "profiles/standard_grpo/summary.json",
    "tables/review_revision/control-summary.json":
        "profiles/review_revision/control-summary.json",
    "paper/RLVRAMBench_IEEE_Access_submission.tex":
        "paper/submission/ieee_access_submission/RLVRAMBench_IEEE_Access_submission.tex",
    "paper/RLVRAMBench_IEEE_Access_submission.pdf":
        "paper/submission/ieee_access_submission/RLVRAMBench_IEEE_Access_submission.pdf",
    "paper/RLVRAMBench_supplement.tex":
        "paper/submission/ieee_access_submission/RLVRAMBench_supplement.tex",
    "paper/RLVRAMBench_supplement.pdf":
        "paper/submission/ieee_access_submission/RLVRAMBench_supplement.pdf",
    "paper/RLVRAMBench_IEEE_Access_source.zip":
        "paper/submission/RLVRAMBench_IEEE_Access_source.zip",
    "paper/figures/standard_grpo_boundary.pdf":
        "paper/figures/standard_grpo_boundary.pdf",
    "verification/admission-source-package.json":
        "paper/submission/ieee_access_submission/export_verification.json",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offload_manifest(tree):
    with tree.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["path", "sha256", "size_bytes"])
        writer.writeheader()
        for path in sorted(tree.root.rglob("*")):
            if path.is_file():
                writer.writerow(dict(path=path.relative_to(tree.root).as_posix(),
                                     **staging.metadata(path)))


@pytest.fixture
def offload_release_tree(tmp_path, monkeypatch):
    root, code = tmp_path / "offload-release", tmp_path / "offload-code"
    code.mkdir()
    for name in staging.OFFLOAD_WRAPPER_FILES:
        (code / name).write_text("# offload wrapper " + name + "\n")
        archived = root / "reproduction_sources/driver" / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes((code / name).read_bytes())
    container = root / "reproduction_sources/offload_factorial"
    source = container / "source"
    design = "benchmark/offload_factorial"
    for name in ("freeze.json", "protocol.json", "matrix.csv", "predictions.json"):
        write_json(source / design / name, {"synthetic": name})
        write_json(root / design / name, {"synthetic": name})
    monkeypatch.setattr(staging, "OFFLOAD_FREEZE_SHA256", digest(source / design / "freeze.json"))
    helper = container / "snapshot_offload_factorial.py"
    helper.write_text("# synthetic export verifier\n")
    files = {path.relative_to(source).as_posix(): staging.metadata(path)
             for path in source.rglob("*") if path.is_file()}
    write_json(container / "source-export-report.json", dict(
        schema_version="offload-source-export-1", status="verified_git_free_cpu_export",
        source_commit=staging.OFFLOAD_SOURCE_COMMIT, freeze_sha256=staging.OFFLOAD_FREEZE_SHA256,
        source_directory="source", files=files, source_file_count=len(files),
        source_total_bytes=sum(item["size_bytes"] for item in files.values()),
        verification_code=dict(sha256=digest(helper)),
    ))
    write_json(root / design / "allocation_accounting.json", [])
    output = root / design / "results"
    output.mkdir()
    for name in staging.OFFLOAD_OUTPUTS - {"summary.json"}:
        (output / name).write_text("peak_mib\n37\n")
    accounting = dict(allocations=1, whole_allocation_gpu_seconds=120)
    write_json(output / "summary.json", dict(
        peak_mib=37, raw_root="/original/offload/raw", source_snapshot="/original/offload/source",
        allocation_accounting=accounting,
    ))
    release = staging.offload_release_fields(root, code)
    contract = release["offload_reproduction"]
    report = dict(
        **{flag: True for flag in staging.OFFLOAD_RECONSTRUCTION_FLAGS},
        **{key: contract[key] for key in (
            "source_commit", "freeze_sha256", "source_tree_sha256", "source_export_receipt_sha256",
            "source_export_helper_sha256", "allocation_accounting_sha256", "output_sha256")},
        panel_files_matched=6, fresh_refit_performed=False, prediction_rows_replayed=288,
        whole_allocation_accounting=accounting,
        relocated_provenance={
            "raw_root": {"published": "/original/offload/raw", "reconstructed": "/new/offload/raw"},
            "source_snapshot": {"published": "/original/offload/source", "reconstructed": "/new/offload/source"},
        },
    )
    tree = SimpleNamespace(
        root=root, code=code, source=source, container=container,
        manifest=tmp_path / "offload-manifest.csv", release=release,
        result=dict(offload_reproduction=report,
                    offload_reproduction_contract_sha256=staging.canonical_digest(contract)),
    )
    offload_manifest(tree)
    return tree


def test_offload_preparation_and_finalization_bind_six_outputs(offload_release_tree):
    tree = offload_release_tree
    assert set(tree.release["offload_reproduction"]["output_sha256"]) == staging.OFFLOAD_OUTPUTS
    staging.verify_offload_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


def test_old_release_needs_no_offload_metadata(tmp_path):
    assert staging.offload_release_fields(tmp_path, tmp_path) == {}
    staging.verify_offload_reconstruction(tmp_path, {}, {}, tmp_path / "absent")


@pytest.mark.parametrize("mutation", [
    "missing_results", "extra_results", "source_change", "receipt_change",
    "helper_change", "wrong_commit", "wrong_freeze", "public_protocol", "wrapper_change",
    "linked_source", "missing_accounting",
])
def test_offload_prepare_rejects_incomplete_or_changed_export(offload_release_tree, mutation):
    tree = offload_release_tree
    if mutation == "missing_results":
        (tree.root / "benchmark/offload_factorial/results/slots.csv").unlink()
    elif mutation == "extra_results":
        (tree.root / "benchmark/offload_factorial/results/extra.csv").write_text("extra\n")
    elif mutation == "source_change":
        (tree.source / "unsealed.py").write_text("# changed\n")
    elif mutation in ("receipt_change", "wrong_commit", "wrong_freeze"):
        path = tree.container / "source-export-report.json"
        value = json.loads(path.read_text())
        key = {"receipt_change": "status", "wrong_commit": "source_commit", "wrong_freeze": "freeze_sha256"}[mutation]
        value[key] = "changed"
        write_json(path, value)
    elif mutation == "helper_change":
        (tree.container / "snapshot_offload_factorial.py").write_text("# changed\n")
    elif mutation == "public_protocol":
        write_json(tree.root / "benchmark/offload_factorial/protocol.json", {"changed": True})
    elif mutation == "wrapper_change":
        (tree.code / "offload_reproduction.py").write_text("# changed\n")
    elif mutation == "missing_accounting":
        (tree.root / "benchmark/offload_factorial/allocation_accounting.json").unlink()
    else:
        (tree.source / "linked.py").symlink_to(tree.container / "snapshot_offload_factorial.py")
    with pytest.raises((ValueError, FileNotFoundError)):
        staging.offload_release_fields(tree.root, tree.code)


@pytest.mark.parametrize("flag", staging.OFFLOAD_RECONSTRUCTION_FLAGS)
def test_offload_finalization_requires_each_independent_gate(offload_release_tree, flag):
    tree = offload_release_tree
    tree.result["offload_reproduction"][flag] = False
    with pytest.raises(ValueError):
        staging.verify_offload_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


@pytest.mark.parametrize("mutation", [
    "no_contract", "no_report", "wrong_contract", "wrong_schema", "refit", "wrong_count",
    "wrong_prediction_count", "wrong_source", "wrong_receipt", "wrong_accounting",
    "changed_costs", "changed_output", "extra_output", "changed_source_manifest",
    "duplicate_manifest", "wrong_location", "missing_location",
])
def test_offload_finalization_rejects_stale_or_nonindependent_report(offload_release_tree, mutation):
    tree = offload_release_tree
    report = tree.result["offload_reproduction"]
    if mutation == "no_contract":
        tree.release.pop("offload_reproduction")
    elif mutation == "no_report":
        tree.result.pop("offload_reproduction")
    elif mutation == "wrong_contract":
        tree.result["offload_reproduction_contract_sha256"] = "0" * 64
    elif mutation == "wrong_schema":
        tree.release["offload_reproduction"]["schema_version"] = True
    elif mutation == "refit":
        report["fresh_refit_performed"] = True
    elif mutation == "wrong_count":
        report["panel_files_matched"] = 5
    elif mutation == "wrong_prediction_count":
        report["prediction_rows_replayed"] = 287
    elif mutation in ("wrong_source", "wrong_receipt"):
        key = "source_tree_sha256" if mutation == "wrong_source" else "source_export_receipt_sha256"
        report[key] = "0" * 64
    elif mutation == "wrong_accounting":
        write_json(tree.root / "benchmark/offload_factorial/allocation_accounting.json", [1])
    elif mutation == "changed_costs":
        report["whole_allocation_accounting"]["whole_allocation_gpu_seconds"] = 121
    elif mutation == "changed_output":
        (tree.root / "benchmark/offload_factorial/results/slots.csv").write_text("peak_mib\n38\n")
    elif mutation == "extra_output":
        (tree.root / "benchmark/offload_factorial/results/extra.json").write_text("{}\n")
    elif mutation == "changed_source_manifest":
        tree.manifest.write_text(tree.manifest.read_text().replace(
            "reproduction_sources/offload_factorial/source/benchmark/",
            "reproduction_sources/offload_factorial/source/changed/"))
    elif mutation == "duplicate_manifest":
        lines = tree.manifest.read_text().splitlines(keepends=True)
        tree.manifest.write_text("".join(lines + lines[1:2]))
    elif mutation == "wrong_location":
        report["relocated_provenance"]["raw_root"]["published"] = "/wrong/raw/path"
    else:
        report.pop("relocated_provenance")
    with pytest.raises(ValueError):
        staging.verify_offload_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


@pytest.fixture
def replication_release_tree(tmp_path):
    root, code = tmp_path / "replication", tmp_path / "code"
    code.mkdir()
    for name in staging.REPLICATION_WRAPPER_FILES:
        (code / name).write_text("# isolated wrapper\n")
        target = root / "reproduction_sources/driver" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((code / name).read_bytes())
    for name in staging.REPLICATION_PANELS:
        output = root / "benchmark" / name / "results"
        output.mkdir(parents=True)
        for filename in staging.REPLICATION_OUTPUTS[name]:
            (output / filename).write_text("{}\n" if filename.endswith(".json") else "key\n")
        source = root / "reproduction_sources" / name
        if name == "completion_replication":
            for cohort in ("original", "amended"):
                write_json(source / cohort / "benchmark/completion_replication/freeze.json",
                           {"frozen": "before execution"})
            write_json(source / "completion-runtime-amendment-01.json", {"allowed": "remaining"})
            write_json(source / "source-export-report.json", {"verified": True})
            write_json(root / "benchmark/completion_replication/allocation_accounting.json", [])
        else:
            for filename in ("replication_control_execution_seal.json",
                             "replication_control_runtime_amendment.json"):
                write_json(source / "benchmark/replication_control" / filename, {"bound": True})
            write_json(source / "replication_control_base_source/source.json", {"original": True})
            write_json(source / "replication_control_base_source/benchmark/replication_control/"
                       "replication_control_execution_seal.json", {"bound": True})
    release = staging.replication_release_fields(root, code)
    result = {
        "replication_reproduction": {},
        "replication_reproduction_contract_sha256":
            staging.canonical_digest(release["replication_reproduction"]),
    }
    for name, settings in release["replication_reproduction"]["panels"].items():
        report = {
            "panel_files_matched": 4, "archived_source_only": True,
            "source_and_manifest_verified": True, "independent_raw_reconstruction_passed": True,
            "reference_outcomes_hidden": True, "child_namespace_verified": True,
            "runtime_amendment_sha256": settings["runtime_amendment_sha256"],
            "source_tree_sha256": settings["source_tree_sha256"],
        }
        if name == "completion_replication":
            report.update(prediction_replay_passed=True, prediction_replay_outcomes_hidden=True,
                          fresh_refit_performed=False,
                          allocation_accounting_sha256=settings["allocation_accounting_sha256"],
                          source_export_receipt_verified=True,
                          source_export_receipt_sha256=digest(
                              root / "reproduction_sources/completion_replication/source-export-report.json"),
                          collection_cpu_wall_seconds={
                              "published_seconds": 2.0, "reconstructed_seconds": 3.0})
        result["replication_reproduction"][name] = report
    manifest = tmp_path / "replication-manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["path", "sha256", "size_bytes"])
        writer.writeheader()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                writer.writerow({"path": str(path.relative_to(root)), **staging.metadata(path)})
    return SimpleNamespace(root=root, code=code, release=release, result=result, manifest=manifest)


def test_replication_finalization_checks_archived_sources_and_separate_timing(replication_release_tree):
    tree = replication_release_tree
    staging.verify_replication_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


@pytest.mark.parametrize("mutation", [
    "metadata_missing", "panel_missing", "source_digest", "amendment_digest",
    "outcomes_visible", "prediction_visible", "refit", "bad_timing", "timing_names",
    "accounting_changed", "source_manifest_changed",
])
def test_replication_finalization_rejects_incomplete_or_stale_verification(
        replication_release_tree, mutation):
    tree = replication_release_tree
    report = tree.result["replication_reproduction"]["completion_replication"]
    if mutation == "metadata_missing":
        tree.release.clear()
    elif mutation == "panel_missing":
        tree.result["replication_reproduction"].pop("replication_control")
    elif mutation == "source_digest":
        report["source_tree_sha256"] = {"original": "0" * 64, "amended": "0" * 64}
    elif mutation == "amendment_digest":
        report["runtime_amendment_sha256"] = "0" * 64
    elif mutation == "outcomes_visible":
        report["reference_outcomes_hidden"] = False
    elif mutation == "prediction_visible":
        report["prediction_replay_outcomes_hidden"] = False
    elif mutation == "refit":
        report["fresh_refit_performed"] = True
    elif mutation == "bad_timing":
        report["collection_cpu_wall_seconds"]["reconstructed_seconds"] = float("nan")
    elif mutation == "timing_names":
        report["collection_cpu_wall_seconds"] = {"a": 1, "b": 2}
    elif mutation == "accounting_changed":
        write_json(tree.root / "benchmark/completion_replication/allocation_accounting.json", [1])
    else:
        tree.manifest.write_text(tree.manifest.read_text().replace("source.json", "changed.json"))
    with pytest.raises(ValueError):
        staging.verify_replication_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


def test_replication_staging_does_not_silently_omit_unfinished_panels(tmp_path):
    (tmp_path / "benchmark/completion_replication").mkdir(parents=True)
    with pytest.raises(ValueError, match="Complete replication"):
        staging.replication_release_fields(tmp_path, tmp_path / "code")


def test_replication_source_digest_rejects_links(replication_release_tree, tmp_path):
    tree = replication_release_tree
    source = tree.root / "reproduction_sources/replication_control"
    external = tmp_path / "external.py"
    external.write_text("# external\n")
    (source / "linked.py").symlink_to(external)
    with pytest.raises(ValueError, match="Nonregular"):
        staging.replication_release_fields(tree.root, tree.code)


@pytest.mark.parametrize("mutation", [
    "empty_cohorts", "amendment", "freeze", "seal", "empty_wrapper",
    "wrong_wrapper", "receipt", "changed_contract", "control_base_only",
])
def test_replication_checks_joint_release_report_mutations_against_manifest(
        replication_release_tree, mutation):
    tree = replication_release_tree
    contract = tree.release["replication_reproduction"]
    settings = contract["panels"]["completion_replication"]
    report = tree.result["replication_reproduction"]["completion_replication"]
    if mutation == "empty_cohorts":
        settings["source_tree_sha256"] = report["source_tree_sha256"] = {}
    elif mutation == "amendment":
        settings["runtime_amendment_sha256"] = report["runtime_amendment_sha256"] = "0" * 64
    elif mutation == "freeze":
        settings["freeze_sha256"] = "0" * 64
    elif mutation == "seal":
        contract["panels"]["replication_control"]["execution_seal_sha256"] = "0" * 64
    elif mutation == "empty_wrapper":
        contract["code_sha256"] = {}
    elif mutation == "wrong_wrapper":
        contract["code_sha256"]["reproduce.py"] = "0" * 64
    elif mutation == "receipt":
        report["source_export_receipt_sha256"] = "0" * 64
    elif mutation == "changed_contract":
        contract["code_sha256"]["reproduce.py"] = "0" * 64
    else:
        path = "replication_control_base_source/benchmark/replication_control/"
        tree.manifest.write_text(tree.manifest.read_text().replace(
            path + "replication_control_execution_seal.json", path + "wrong-seal.json"))
    # Even an internally consistent release/report must not override the
    # authenticated archive. The separate changed-contract case keeps the
    # actual run's old contract digest to exercise that independent gate.
    if mutation != "changed_contract":
        tree.result["replication_reproduction_contract_sha256"] = staging.canonical_digest(contract)
    with pytest.raises(ValueError):
        staging.verify_replication_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


def add_control_accounting(tree):
    from memory_tuner.capture_allocation_accounting import capture
    for name in staging.REPLICATION_ACCOUNTING_CODE_FILES:
        path = tree.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# accounting source: " + name + "\n")
    text = "\n".join(
        f"123_0{suffix}|123{suffix}|{name}|FAILED|2:0|"
        "2026-09-16T10:00:00|2026-09-16T10:01:00|60|cpu=32,gres/gpu=2|node|None"
        for suffix, name in (("", "control"), (".batch", "batch"))
    )
    records = capture(text, {"123": {
        "pair_id": "control-a", "pair_manifest_sha256": "a" * 64, "pair_ledger_sha256": "b" * 64,
    }})
    path = tree.root / "benchmark/replication_control/allocation_accounting.json"
    write_json(path, records)
    tree.release = staging.replication_release_fields(tree.root, tree.code)
    settings = tree.release["replication_reproduction"]["panels"]["replication_control"]
    tree.result["replication_reproduction_contract_sha256"] = staging.canonical_digest(
        tree.release["replication_reproduction"])
    tree.result["replication_reproduction"]["replication_control"].update(
        allocation_accounting_sha256=settings["allocation_accounting_sha256"],
        accounting_code_sha256=settings["accounting_code_sha256"],
        whole_allocation_accounting_reconstruction_passed=True,
        whole_allocation_accounting={
            "schema_version": 1, "allocation_count": 1, "whole_allocation_gpu_seconds": 120,
            "allocations": [{"block_id": "control-a", "job_id": "123", "scheduler_state": "FAILED",
                             "whole_allocation_gpu_seconds": 120, "batch_step_gpu_seconds": 120}],
        },
    )
    with tree.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["path", "sha256", "size_bytes"])
        writer.writeheader()
        for path in sorted(tree.root.rglob("*")):
            if path.is_file():
                writer.writerow({"path": str(path.relative_to(tree.root)), **staging.metadata(path)})


def test_replication_finalization_rechecks_separate_control_costs(replication_release_tree):
    tree = replication_release_tree
    add_control_accounting(tree)
    staging.verify_replication_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


@pytest.mark.parametrize("mutation", [
    "omitted_metadata", "helper_digest", "double_counted", "wrong_job", "input_changed", "flag_missing",
])
def test_replication_finalization_rejects_unbound_or_wrong_control_costs(
        replication_release_tree, mutation):
    tree = replication_release_tree
    add_control_accounting(tree)
    contract = tree.release["replication_reproduction"]
    settings = contract["panels"]["replication_control"]
    report = tree.result["replication_reproduction"]["replication_control"]
    if mutation == "omitted_metadata":
        settings.pop("allocation_accounting_sha256")
        settings.pop("accounting_code_sha256")
    elif mutation == "helper_digest":
        settings["accounting_code_sha256"]["memory_tuner/replication_control_accounting.py"] = "0" * 64
    elif mutation == "double_counted":
        report["whole_allocation_accounting"]["whole_allocation_gpu_seconds"] = 240
    elif mutation == "wrong_job":
        report["whole_allocation_accounting"]["allocations"][0]["job_id"] = "456"
    elif mutation == "input_changed":
        write_json(tree.root / "benchmark/replication_control/allocation_accounting.json", [])
    else:
        report["whole_allocation_accounting_reconstruction_passed"] = False
    tree.result["replication_reproduction_contract_sha256"] = staging.canonical_digest(contract)
    with pytest.raises(ValueError):
        staging.verify_replication_reconstruction(tree.root, tree.release, tree.result, tree.manifest)


@pytest.fixture
def release_tree(tmp_path):
    stage = tmp_path / "stage"
    archives = stage / "releases"
    archives.mkdir(parents=True)
    entries = []
    for name, content in (
        ("evidence.tar.zst", b"synthetic evidence archive"),
        ("submission.zip", b"synthetic submission archive"),
    ):
        path = archives / name
        path.write_bytes(content)
        entries.append({
            "path": str(path), "sha256": digest(path),
            "size_bytes": len(content),
        })
    manifest = (tmp_path / "reconstruction/reference-profiles/benchmark/"
                "publication-artifact-manifest.csv")
    manifest.parent.mkdir(parents=True)
    records = []
    for public_name, original in PUBLIC_PATHS.items():
        path = stage / public_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("frozen: " + original + "\n").encode())
        records.append({
            "path": original, "category": "test",
            "sha256": digest(path), "size_bytes": path.stat().st_size,
        })
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["path", "category", "sha256", "size_bytes"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    built = {
        "tag": "regression-release", "archives": entries,
        "review_members": 31, "manifest_entries": len(records),
    }
    built_path = archives / "RLVRAMBench-release-regression-release.json"
    write_json(built_path, built)
    release = {
        "release": built["tag"],
        "archive": {**entries[0], "path": "releases/evidence.tar.zst"},
        "archive_members": 31, "manifest_entries": len(records),
        "expected_derived_files": 40, "expected_review_derived_files": 27,
        "expected_curated_benchmark_files": 14,
        "expected_decision_derived_files": 11,
        "decision_validated_memory_outcomes": 48,
    }
    release_path = stage / "release.json"
    write_json(release_path, release)
    result = {
        "status": "passed", "archive_sha256": entries[0]["sha256"],
        "archive_members": 31, "manifest_entries_verified": len(records),
        "derived_csv_json_files_matched": 40,
        "revised_csv_json_files_matched": 27,
        "curated_benchmark_files_matched": 14,
        "decision_panel_files_matched": 11,
        "decision_validated_memory_outcomes": 48,
        "original_project_and_reference_tables_hidden": True,
        "decision_reference_outcomes_hidden_during_analysis": True,
        "curated_references_hidden_during_analysis": True,
        "child_namespace_reference_inaccessibility_checked": True,
        "evidence_manifest_sha256": digest(manifest),
    }
    report = tmp_path / "reconstruction/verification.json"
    write_json(report, result)
    return SimpleNamespace(
        stage=stage, archives=archives, built=built, built_path=built_path,
        release=release, release_path=release_path, result=result,
        report=report, manifest=manifest,
    )


def test_finalize_accepts_matching_release_and_hashes_exact_public_payload(release_tree):
    tree = release_tree
    staging.finalize(tree.stage, tree.report)
    manifest = json.loads((tree.stage / "upload-manifest.json").read_text())
    files = {str(p.relative_to(tree.stage)) for p in tree.stage.rglob("*")
             if p.is_file() and p.name != "upload-manifest.json"}
    assert manifest["algorithm"] == "sha256"
    assert set(manifest["files"]) == files
    for name, record in manifest["files"].items():
        path = tree.stage / name
        assert record == {"sha256": digest(path), "size_bytes": path.stat().st_size}
    assert (tree.stage / "verification/admission-reproduction.json").read_bytes() == (
        tree.report.read_bytes())


@pytest.mark.parametrize("field,bad_value", [
    ("status", "failed"),
    ("archive_sha256", "0" * 64),
    ("archive_members", 30),
    ("manifest_entries_verified", 0),
    ("derived_csv_json_files_matched", 39),
    ("revised_csv_json_files_matched", 26),
    ("curated_benchmark_files_matched", 0),
    ("decision_panel_files_matched", 10),
    ("decision_validated_memory_outcomes", 47),
    ("original_project_and_reference_tables_hidden", False),
    ("decision_reference_outcomes_hidden_during_analysis", False),
    ("curated_references_hidden_during_analysis", False),
    ("child_namespace_reference_inaccessibility_checked", False),
])
def test_finalize_rejects_wrong_report_before_publication(release_tree, field, bad_value):
    tree = release_tree
    tree.result[field] = bad_value
    write_json(tree.report, tree.result)
    with pytest.raises(ValueError, match="reconstruction"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()
    assert not (tree.stage / "verification/admission-reproduction.json").exists()


def test_finalize_rejects_report_from_before_manifest_and_namespace_checks(release_tree):
    tree = release_tree
    for field in ("archive_sha256", "evidence_manifest_sha256",
                  "child_namespace_reference_inaccessibility_checked"):
        tree.result.pop(field)
    write_json(tree.report, tree.result)
    with pytest.raises((KeyError, ValueError)):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


@pytest.mark.parametrize("field", ["review_members", "manifest_entries"])
def test_finalize_rejects_build_counts_even_if_report_and_release_agree(release_tree, field):
    tree = release_tree
    tree.built[field] += 1
    write_json(tree.built_path, tree.built)
    with pytest.raises(ValueError, match="reconstruction"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


@pytest.mark.parametrize("name", ["evidence.tar.zst", "submission.zip"])
def test_finalize_rehashes_both_staged_archives(release_tree, name):
    tree = release_tree
    path = tree.archives / name
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError, match="Archive differs"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


def test_finalize_checks_release_archive_digest_independently_of_build(release_tree):
    tree = release_tree
    tree.release["archive"]["sha256"] = "f" * 64
    write_json(tree.release_path, tree.release)
    with pytest.raises(ValueError, match="release.json"):
        staging.finalize(tree.stage, tree.report)


def test_finalize_rejects_manifest_changed_after_reconstruction(release_tree):
    tree = release_tree
    tree.manifest.write_bytes(tree.manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest changed"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


@pytest.mark.parametrize("name", list(PUBLIC_PATHS))
def test_finalize_rejects_changed_mapped_payload(release_tree, name):
    tree = release_tree
    path = tree.stage / name
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError, match="Loose publication file differs"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


def test_finalize_rejects_unmapped_extra_payload(release_tree):
    tree = release_tree
    (tree.stage / "unreviewed.json").write_text("{}")
    with pytest.raises(ValueError, match="Unmapped publication payload"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


@pytest.mark.parametrize("name", [
    "benchmark/decision/results/candidate_results.csv",
    "paper/RLVRAMBench_IEEE_Access_submission.pdf",
])
def test_finalize_rejects_required_loose_file_removed_after_preparation(release_tree, name):
    tree = release_tree
    (tree.stage / name).unlink()
    with pytest.raises(ValueError):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


@pytest.mark.parametrize("archive_name", ["evidence.tar.zst", "submission.zip"])
def test_prepare_rejects_corrupt_archive_before_changing_code_or_stage(
        release_tree, tmp_path, monkeypatch, archive_name):
    tree = release_tree
    root, code, destination = (tmp_path / name for name in ("source", "code", "new-stage"))
    write_json(root / "benchmark/decision/results/collection_summary.json",
               {"planned_slots": 48})
    write_json(code / "release.json", {"release": "previous"})
    previous = (code / "release.json").read_bytes()
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *a, **k: "a" * 40)
    path = tree.archives / archive_name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Archive differs"):
        staging.prepare(root, code, destination, tree.built_path, "a" * 40)
    assert not destination.exists()
    assert (code / "release.json").read_bytes() == previous


def test_export_verification_json_is_included_in_evidence_manifest(tmp_path):
    path = tmp_path / "paper/submission/ieee_access_submission/export_verification.json"
    write_json(path, {"source_bundle_verified_after_extraction": True})
    files, missing = collect_files(tmp_path)
    assert not missing
    assert path.resolve() in files


def estimation_fixture(root, *, eligible=29, resolved=8, accounting=True):
    """Synthetic partial collection; no actual study records are consulted."""
    directory = root / "benchmark/estimation"
    output = directory / "results"
    protocol = {
        "protocol_version": "estimation-1.0", "planned_processes": 36,
        "planned_pairs": 18, "target_configurations": 12,
    }
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "prediction_freeze.json", {
        "protocol_sha256": digest(directory / "protocol.json"),
    })
    matrix = directory / "matrix.csv"
    matrix.write_text("pair_id,period\n" + "".join(
        f"pair-{index},{period}\n" for index in range(18) for period in (1, 2)))
    commits = {f"pair-{index}": ("a" if index < 2 else "b") * 40
               for index in range(18)}
    write_json(directory / "execution_commits.json", commits)
    summary = {
        "schema_version": 1, "planned_processes": 36, "planned_pairs": 18,
        "planned_configurations": 12, "eligible_processes": eligible,
        "completed_processes": min(22, eligible),
        "unresolved_processes": 36 - eligible,
        "known_launched_processes": 34, "unknown_launch_status_processes": 1,
        "resolved_configurations": resolved, "pair_errors": {"pair-0": "setup failure"},
    }
    write_json(output / "summary.json", summary)
    write_json(output / "evaluation.json", {
        "planned_configurations": 12, "resolved_configurations": resolved,
        "unresolved_configurations": 12 - resolved,
    })
    provenance = {
        "protocol_sha256": digest(directory / "protocol.json"),
        "prediction_seal_sha256": digest(directory / "prediction_freeze.json"),
        "execution_commits_expected": commits,
        "execution_commits_file_sha256": digest(directory / "execution_commits.json"),
    }
    if accounting:
        write_json(directory / "allocation_accounting.json", [])
        provenance["allocation_accounting_sha256"] = digest(directory / "allocation_accounting.json")
    write_json(output / "provenance.json", provenance)
    (output / "processes.csv").write_text("experiment_id,state\npartial,unresolved\n")
    write_json(output / "output_manifest.json", {
        p.name: staging.metadata(p) for p in output.iterdir() if p.is_file()
    })
    return directory


@pytest.fixture
def estimation_release_tree(release_tree):
    tree = release_tree
    estimation_fixture(tree.stage)
    fields = staging.estimation_release_fields(tree.stage)
    tree.release.update(fields)
    write_json(tree.release_path, tree.release)
    tree.result.update({
        "estimation_panel_files_matched": fields["expected_estimation_derived_files"],
        "estimation_counts": deepcopy(fields["estimation_counts"]),
        "estimation_prediction_seal_sha256": fields["estimation_prediction_seal_sha256"],
        "estimation_execution_commits": deepcopy(fields["estimation_execution_commits"]),
        "estimation_fresh_refit_compared": False,
        **{key: True for key in staging.ESTIMATION_RECONSTRUCTION_FLAGS},
    })
    with tree.manifest.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    for path in (tree.stage / "benchmark/estimation").rglob("*"):
        if path.is_file():
            records.append({
                "path": str(path.relative_to(tree.stage)), "category": "estimation",
                **staging.metadata(path),
            })
    with tree.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["path", "category", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(records)
    tree.built["manifest_entries"] = tree.release["manifest_entries"] = len(records)
    tree.result["manifest_entries_verified"] = len(records)
    tree.result["evidence_manifest_sha256"] = digest(tree.manifest)
    write_json(tree.built_path, tree.built)
    write_json(tree.release_path, tree.release)
    write_json(tree.report, tree.result)
    return tree


def test_estimation_metadata_comes_from_partial_collector_and_dynamic_file_count(tmp_path):
    estimation_fixture(tmp_path)
    fields = staging.estimation_release_fields(tmp_path)
    assert fields["expected_estimation_derived_files"] == 5
    assert fields["estimation_counts"]["eligible_processes"] == 29
    assert fields["estimation_counts"]["unresolved_processes"] == 7
    assert fields["estimation_counts"]["resolved_configurations"] == 8
    assert fields["estimation_counts"]["unresolved_configurations"] == 4
    assert set(fields["estimation_execution_commits"].values()) == {"a" * 40, "b" * 40}


def test_estimation_metadata_allows_zero_eligible_outcomes_and_no_accounting(tmp_path):
    estimation_fixture(tmp_path, eligible=0, resolved=0, accounting=False)
    fields = staging.estimation_release_fields(tmp_path)
    assert fields["estimation_counts"]["eligible_processes"] == 0
    assert fields["estimation_counts"]["resolved_configurations"] == 0
    assert fields["estimation_allocation_accounting_sha256"] is None


@pytest.mark.parametrize("file", ["summary.json", "provenance.json", "processes.csv"])
def test_estimation_metadata_rejects_changed_collector_outputs(tmp_path, file):
    directory = estimation_fixture(tmp_path)
    path = directory / "results" / file
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="collector output differs"):
        staging.estimation_release_fields(tmp_path)


@pytest.mark.parametrize("file", [
    "prediction_freeze.json", "execution_commits.json", "allocation_accounting.json",
])
def test_estimation_metadata_rejects_changed_frozen_or_collector_inputs(tmp_path, file):
    directory = estimation_fixture(tmp_path)
    path = directory / file
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Estimation"):
        staging.estimation_release_fields(tmp_path)


def test_estimation_metadata_rejects_partial_output_publication(tmp_path):
    directory = estimation_fixture(tmp_path)
    (directory / "results/output_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        staging.estimation_release_fields(tmp_path)


def test_finalize_accepts_partial_estimation_with_separate_refit_status(estimation_release_tree):
    tree = estimation_release_tree
    staging.finalize(tree.stage, tree.report)
    assert (tree.stage / "upload-manifest.json").is_file()
    published = json.loads((tree.stage / "verification/admission-reproduction.json").read_text())
    assert published["estimation_counts"]["eligible_processes"] == 29
    assert published["estimation_fresh_refit_compared"] is False


@pytest.mark.parametrize("field,bad_value", [
    *[(key, False) for key in staging.ESTIMATION_RECONSTRUCTION_FLAGS],
    ("estimation_independent_raw_reconstruction_passed", 1),
    ("estimation_panel_files_matched", 13),
    ("estimation_prediction_seal_sha256", "0" * 64),
    ("estimation_execution_commits", {"pair-0": "c" * 40}),
    ("estimation_counts", {"eligible_processes": 36}),
    ("estimation_fresh_refit_compared", "not run"),
])
def test_finalize_rejects_incomplete_estimation_gates(estimation_release_tree, field, bad_value):
    tree = estimation_release_tree
    tree.result[field] = bad_value
    write_json(tree.report, tree.result)
    with pytest.raises(ValueError, match="[Ee]stimation"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()
    assert not (tree.stage / "verification/admission-reproduction.json").exists()


def test_finalize_cannot_disable_estimation_by_removing_release_fields(estimation_release_tree):
    tree = estimation_release_tree
    for key in list(tree.release):
        if key.startswith("estimation_") or key == "expected_estimation_derived_files":
            del tree.release[key]
    write_json(tree.release_path, tree.release)
    with pytest.raises(ValueError, match="Estimation release metadata"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()


def test_finalize_rechecks_estimation_counts_even_if_release_and_report_agree(estimation_release_tree):
    tree = estimation_release_tree
    tree.release["estimation_counts"]["eligible_processes"] = 36
    tree.result["estimation_counts"]["eligible_processes"] = 36
    write_json(tree.release_path, tree.release)
    write_json(tree.report, tree.result)
    with pytest.raises(ValueError, match="Estimation release metadata"):
        staging.finalize(tree.stage, tree.report)


@pytest.mark.parametrize("with_estimation,with_host,with_offload", [
    (False, False, False), (True, False, False), (True, True, False),
    (False, False, True), (True, True, True),
])
def test_prepare_preserves_old_workflow_and_copies_optional_source_closure(
        release_tree, offload_release_tree, tmp_path, monkeypatch,
        with_estimation, with_host, with_offload):
    tree = release_tree
    root, code, destination = (tmp_path / name for name in ("source", "code", "new-stage"))
    for public_name, original in PUBLIC_PATHS.items():
        source = tree.stage / public_name
        target = root / original
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for stem in ("standard_grpo_boundary", "standard_grpo_mechanisms",
                 "standard_grpo_temporal", "review_instrumentation_batch"):
        for suffix in (".pdf", ".png"):
            path = root / "paper/figures" / (stem + suffix)
            path.write_bytes(b"synthetic figure")
    write_json(root / "benchmark/decision/results/collection_summary.json", {
        "planned_slots": 48, "recorded_attempts": 49, "validated_memory_outcomes": 47,
    })
    names = {"benchmark.py"} | staging.ADMISSION_SOURCE_FILES
    if with_estimation:
        estimation_fixture(root)
        names |= staging.ESTIMATION_SOURCE_FILES
    if with_host:
        host_capacity_fixture(root)
        names |= staging.HOST_CAPACITY_SOURCE_FILES
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source closure: " + name)
    if with_offload:
        shutil.copytree(offload_release_tree.root, root, dirs_exist_ok=True)
        shutil.copytree(offload_release_tree.code, code, dirs_exist_ok=True)
    write_json(code / "release.json", {
        "release": "previous", "host_capacity_counts": {"eligible_processes": 999},
        "expected_host_capacity_derived_files": 999,
        "offload_reproduction": {"stale_previous_release": True},
    })
    write_json(code / "source-provenance.json", {"files": {"benchmark.py": {}}})
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *a, **k: "a" * 40)
    staging.prepare(root, code, destination, tree.built_path, "a" * 40)
    release = json.loads((destination / "release.json").read_text())
    provenance = json.loads((code / "source-provenance.json").read_text())
    assert release["prior_immutable_release_preserved"] == "previous"
    assert set(provenance["files"]) == names
    for name in names:
        assert (code / name).read_bytes() == (root / name).read_bytes()
        assert provenance["files"][name] == staging.metadata(root / name)
    if with_estimation:
        assert release["expected_estimation_derived_files"] == 5
        assert release["estimation_counts"]["eligible_processes"] == 29
        assert (code / "benchmark/estimation/execution_commits.json").is_file()
    else:
        assert "expected_estimation_derived_files" not in release
        assert not (code / "memory_tuner/estimation_baselines.py").exists()
    if with_host:
        assert release["expected_host_capacity_derived_files"] == 13
        assert release["host_capacity_counts"]["eligible_processes"] == 5
        assert release["host_capacity_counts"]["planned_processes"] == 18
        assert (code / "benchmark/host_capacity/study_freeze.json").is_file()
        assert (code / "memory_tuner/capture_allocation_accounting.py").is_file()
    else:
        assert "expected_host_capacity_derived_files" not in release
        assert "host_capacity_counts" not in release
    if with_offload:
        assert release["offload_reproduction"] == offload_release_tree.release["offload_reproduction"]
        assert (code / "benchmark/offload_factorial/results/prediction_scores.csv").is_file()
    else:
        assert "offload_reproduction" not in release


def test_finalize_cannot_skip_the_offload_gate(release_tree, monkeypatch):
    def reject(*args):
        raise ValueError("offload gate deliberately failed")
    monkeypatch.setattr(staging, "verify_offload_reconstruction", reject)
    with pytest.raises(ValueError, match="offload gate deliberately failed"):
        staging.finalize(release_tree.stage, release_tree.report)
    assert not (release_tree.stage / "upload-manifest.json").exists()


def host_capacity_fixture(root, *, eligible=5, resolved=0):
    """Complete output schema with deliberately partial synthetic eligibility."""
    if not (root / "benchmark/estimation/results/summary.json").exists():
        estimation_fixture(root)
    original = staging.estimation_release_fields(root)
    directory = root / "benchmark/host_capacity"
    output = directory / "results"
    protocol = {
        "protocol_version": "host-capacity-1.0", "group": "estimation_host_capacity",
        "planned_processes": 18, "planned_pairs": 9, "target_configurations": 6,
        "original_protocol_sha256": original["estimation_protocol_sha256"],
        "original_prediction_seal_sha256": original["estimation_prediction_seal_sha256"],
    }
    write_json(directory / "protocol.json", protocol)
    (directory / "matrix.csv").write_text("pair_id,period,configuration_id\n" + "".join(
        f"host-{i},{period},host-config-{i // 3}-{period}\n"
        for i in range(9) for period in (1, 2)))
    (directory / "targets.csv").write_text("configuration_id\n" + "".join(
        f"host-config-{i}-{period}\n" for i in range(3) for period in (1, 2)))
    write_json(directory / "predictions.json", [])
    freeze = {
        "protocol_sha256": digest(directory / "protocol.json"),
        "original_prediction_seal_sha256": original["estimation_prediction_seal_sha256"],
        "frozen_at_utc": "2026-09-15T18:52:00+00:00",
        "files_sha256": {name: digest(directory / name)
                         for name in ("matrix.csv", "targets.csv", "predictions.json")},
    }
    write_json(directory / "study_freeze.json", freeze)
    commits = {f"host-{i}": "d" * 40 for i in range(9)}
    write_json(directory / "execution_commits.json", commits)
    write_json(directory / "allocation_accounting.json", [])
    summary = {
        "planned_processes": 18, "planned_pairs": 9, "planned_configurations": 6,
        "eligible_processes": eligible, "completed_processes": 0,
        "unresolved_processes": 18 - eligible, "known_launched_processes": 16,
        "unknown_launch_status_processes": 1, "resolved_configurations": resolved,
    }
    values = {
        "summary": summary,
        "evaluation": {"planned_configurations": 6, "resolved_configurations": resolved,
                       "unresolved_configurations": 6 - resolved},
        "provenance": {
            "protocol_sha256": digest(directory / "protocol.json"),
            "study_freeze_sha256": digest(directory / "study_freeze.json"),
            "study_frozen_at_utc": freeze["frozen_at_utc"],
            "original_protocol_sha256": original["estimation_protocol_sha256"],
            "original_prediction_seal_sha256": original["estimation_prediction_seal_sha256"],
            "prediction_seal_sha256": original["estimation_prediction_seal_sha256"],
            "execution_commits_expected": commits,
            "execution_commits_file_sha256": digest(directory / "execution_commits.json"),
            "allocation_accounting_sha256": digest(directory / "allocation_accounting.json"),
        },
        "processes": [], "configurations": [], "configurations_flat": [], "pairs": [], "costs": {},
    }
    for name, value in values.items():
        write_json(output / (name + ".json"), value)
    for name in ("processes", "configurations", "configurations_flat", "pairs"):
        (output / (name + ".csv")).write_text("configuration_id,state\n")
    write_json(output / "output_manifest.json", {
        p.name: staging.metadata(p) for p in output.iterdir() if p.is_file()
    })
    return directory


@pytest.fixture
def host_release_tree(estimation_release_tree):
    tree = estimation_release_tree
    host_capacity_fixture(tree.stage)
    fields = staging.host_capacity_release_fields(tree.stage)
    tree.release.update(fields)
    tree.result.update({key: value for key, value in fields.items()
                        if key != "expected_host_capacity_derived_files"})
    tree.result.update({
        "host_capacity_panel_files_matched": fields["expected_host_capacity_derived_files"],
        "host_capacity_fresh_refit_performed": False,
        **{key: True for key in staging.HOST_CAPACITY_RECONSTRUCTION_FLAGS},
    })
    with tree.manifest.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    for path in (tree.stage / "benchmark/host_capacity").rglob("*"):
        if path.is_file():
            records.append({"path": str(path.relative_to(tree.stage)), "category": "host_capacity",
                            **staging.metadata(path)})
    with tree.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["path", "category", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(records)
    tree.built["manifest_entries"] = tree.release["manifest_entries"] = len(records)
    tree.result["manifest_entries_verified"] = len(records)
    tree.result["evidence_manifest_sha256"] = digest(tree.manifest)
    for path, data in ((tree.built_path, tree.built), (tree.release_path, tree.release),
                       (tree.report, tree.result)):
        write_json(path, data)
    return tree


@pytest.mark.parametrize("eligible,resolved", [(0, 0), (5, 0), (18, 6)])
def test_host_metadata_uses_actual_partial_counts_not_the_original_denominator(tmp_path, eligible, resolved):
    host_capacity_fixture(tmp_path, eligible=eligible, resolved=resolved)
    fields = staging.host_capacity_release_fields(tmp_path)
    assert fields["expected_host_capacity_derived_files"] == 13
    assert fields["host_capacity_counts"]["eligible_processes"] == eligible
    assert fields["host_capacity_counts"]["completed_processes"] == 0
    assert fields["host_capacity_counts"]["unresolved_configurations"] == 6 - resolved
    assert len(fields["host_capacity_execution_commits"]) == 9
    assert fields["host_capacity_prediction_seal_sha256"] == (
        staging.estimation_release_fields(tmp_path)["estimation_prediction_seal_sha256"])


def test_finalize_accepts_separate_partial_host_cohort_with_no_host_fit(host_release_tree):
    tree = host_release_tree
    staging.finalize(tree.stage, tree.report)
    result = json.loads((tree.stage / "verification/admission-reproduction.json").read_text())
    assert result["host_capacity_counts"]["eligible_processes"] == 5
    assert result["estimation_counts"]["eligible_processes"] == 29
    assert result["host_capacity_fresh_refit_performed"] is False


@pytest.mark.parametrize("field,bad_value", [
    *[(key, False) for key in staging.HOST_CAPACITY_RECONSTRUCTION_FLAGS],
    ("host_capacity_independent_raw_reconstruction_passed", 1),
    ("host_capacity_panel_files_matched", 12),
    ("host_capacity_protocol_sha256", "0" * 64),
    ("host_capacity_study_freeze_sha256", "0" * 64),
    ("host_capacity_prediction_seal_sha256", "0" * 64),
    ("host_capacity_original_protocol_sha256", "0" * 64),
    ("host_capacity_execution_commits", {}),
    ("host_capacity_execution_commits_sha256", "0" * 64),
    ("host_capacity_allocation_accounting_sha256", None),
    ("host_capacity_counts", {"eligible_processes": 18}),
    ("host_capacity_fresh_refit_performed", True),
])
def test_finalize_rejects_incomplete_host_gates(host_release_tree, field, bad_value):
    tree = host_release_tree
    tree.result[field] = bad_value
    write_json(tree.report, tree.result)
    with pytest.raises(ValueError, match="[Hh]ost-capacity"):
        staging.finalize(tree.stage, tree.report)
    assert not (tree.stage / "upload-manifest.json").exists()
    assert not (tree.stage / "verification/admission-reproduction.json").exists()


def test_host_cannot_be_disabled_by_dropping_its_release_fields(host_release_tree):
    tree = host_release_tree
    for key in list(tree.release):
        if key.startswith("host_capacity_") or key == "expected_host_capacity_derived_files":
            del tree.release[key]
    write_json(tree.release_path, tree.release)
    with pytest.raises(ValueError, match="Host-capacity release metadata"):
        staging.finalize(tree.stage, tree.report)


@pytest.mark.parametrize("file", [
    "protocol.json", "study_freeze.json", "matrix.csv", "targets.csv", "predictions.json",
    "execution_commits.json", "allocation_accounting.json",
])
def test_host_metadata_rejects_changed_input_bytes(tmp_path, file):
    directory = host_capacity_fixture(tmp_path)
    path = directory / file
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Host-capacity"):
        staging.host_capacity_release_fields(tmp_path)


def test_host_outputs_require_all_thirteen_files_even_if_manifest_is_rehashed(tmp_path):
    directory = host_capacity_fixture(tmp_path) / "results"
    (directory / "costs.json").unlink()
    manifest = json.loads((directory / "output_manifest.json").read_text())
    manifest.pop("costs.json")
    write_json(directory / "output_manifest.json", manifest)
    with pytest.raises(ValueError, match="13 output files"):
        staging.host_capacity_release_fields(tmp_path)


def test_host_needs_an_explicit_accounting_input_and_the_original_cohort(tmp_path):
    directory = host_capacity_fixture(tmp_path)
    (directory / "allocation_accounting.json").unlink()
    with pytest.raises(ValueError, match="accounting"):
        staging.host_capacity_release_fields(tmp_path)
    (tmp_path / "benchmark/estimation/results/summary.json").unlink()
    with pytest.raises(ValueError, match="separately replayed"):
        staging.host_capacity_release_fields(tmp_path)
