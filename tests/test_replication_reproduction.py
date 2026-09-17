"""Synthetic CPU fixtures only: no production replay, models, Slurm or network."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

import reproduce
import replication_reproduction as replication
from test_reproduction_wrapper import reconstruction_tree


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_outputs(panel, directory, peak, raw_root, seconds=1.0):
    directory.mkdir(parents=True, exist_ok=False)
    for name in replication.OUTPUTS[panel]:
        if name.endswith(".json"):
            record = {"peak_mib": peak, "raw_root": str(raw_root)}
            if panel == "completion_replication":
                record[replication.TIMING_FIELD] = seconds
            write_json(directory / name, record)
        else:
            (directory / name).write_text(f"condition,peak_mib\nbase,{peak}\n")


def export_receipt(container, amendment_path):
    amendment = json.loads(amendment_path.read_text())
    trees = {}
    for role in ("original", "amended"):
        root = container / role
        files = {name: dict(sha256=digest, size_bytes=(root / name).stat().st_size)
                 for name, digest in replication.inventory(root).items()}
        protocol = json.loads((root / "benchmark/completion_replication/protocol.json").read_text())
        notices = {name: files[name]["sha256"] for name in (
            "LICENSE-RLVRAMBench", "instrumented_python_packages/verl/LICENSE", "third_party/TransferQueue/LICENSE")}
        trees[role] = dict(
            source_commit=amendment[role + "_source_commit"], files=files, file_count=len(files),
            total_bytes=sum(item["size_bytes"] for item in files.values()),
            runtime_files=len(amendment[role + "_runtime_sha256"]),
            fitting_planning_input_entries=len(protocol["input_sha256"]), freeze_files=8, notices=notices)
    write_json(container / replication.EXPORT_REPORT, dict(
        schema_version="completion-source-export-1", status="verified_git_free_cpu_export",
        amendment_file=amendment_path.name, amendment_sha256=replication.sha256(amendment_path),
        original_freeze_sha256=amendment["original_freeze_sha256"],
        initial_freeze_sha256=amendment["initial_freeze_sha256"], trees=trees,
        cpu_validation=dict(status="passed", loader="amended/memory_tuner/completion_replication_protocol.py",
                            amendment_sha256=replication.sha256(amendment_path),
                            freeze_sha256=amendment["original_freeze_sha256"],
                            planned_slots=1, planned_pairs=1, target_configurations=1, prediction_rows=1,
                            python="synthetic", numpy="synthetic", scipy="synthetic", scikit_learn="synthetic"),
        excludes=["Git metadata", "model weights/cache", "containers", "raw target attempts"],
        notice_provenance="Synthetic notice provenance", raw_root_policy="Explicit synthetic raw root",
        helper_sha256="f" * 64))


def rewrite_manifest(tree):
    tree.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tree.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["path", "size_bytes", "sha256"])
        writer.writeheader()
        for path in sorted(tree.artifact.rglob("*")):
            if path.is_file() and path != tree.manifest:
                writer.writerow(dict(path=path.relative_to(tree.artifact).as_posix(),
                                     size_bytes=path.stat().st_size, sha256=replication.sha256(path)))


def collector_script(panel):
    flags = (
        ("root", "source-root", "protocol-dir", "freeze-sha256", "source-commit",
         "allocation-accounting", "runtime-amendment", "amendment-sha256", "original-source-root", "output")
        if panel == "completion_replication" else
        ("design-dir", "results-root", "seal-sha256", "amendment", "amendment-sha256", "base-source", "output")
    )
    return f"""
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for flag in {flags!r}:
    p.add_argument('--' + flag, required=True)
p.add_argument('--allow-relocation', action='store_true')
a = p.parse_args()
assert Path(__file__).resolve().parents[1] == Path.cwd()
root = Path(a.root) / 'output/completion_replication' if {panel!r} == 'completion_replication' else Path(a.results_root)
peak = json.loads((root / 'raw.json').read_text())['peak_mib']
if {panel!r} == 'completion_replication':
    assert Path(a.original_source_root).is_dir()
    assert Path(a.allocation_accounting).is_file()
    assert not (Path(a.root).parent / 'reference-benchmark/completion_replication/results/summary.json').exists()
out = Path(a.output)
out.mkdir()
for name in {sorted(replication.OUTPUTS[panel])!r}:
    if name.endswith('.json'):
        record = dict(peak_mib=peak, raw_root=str(root))
        if {panel!r} == 'completion_replication':
            record['collection_cpu_wall_seconds'] = 0.125
        (out / name).write_text(json.dumps(record, indent=2) + '\\n')
    else:
        (out / name).write_text(f'condition,peak_mib\\nbase,{{peak}}\\n')
"""


@pytest.fixture
def panel_tree(tmp_path):
    work = tmp_path / "panel-work"
    artifact = work / "artifact"
    archive_sources = artifact / "reproduction_sources"
    completion = archive_sources / "completion_replication"
    control = archive_sources / "replication_control"
    old_control = control / "replication_control_base_source"
    for source in (completion / "original", completion / "amended", control, old_control):
        (source / "memory_tuner").mkdir(parents=True)
        (source / "memory_tuner/__init__.py").write_text("")
    for source in (completion / "original", completion / "amended"):
        directory = source / "benchmark/completion_replication"
        write_json(directory / "protocol.json", {"scientific_input": 42, "input_sha256": {},
                                                 "targets": [{"configuration_id": "synthetic"}]})
        write_json(directory / "predictions.json", [{"peak_mib": 41}])
        (directory / "matrix.csv").write_text("seed,pair_id\n221,synthetic-pair\n")
        write_json(directory / "freeze.json", {
            "files_sha256": {name: replication.sha256(directory / name)
                             for name in ("protocol.json", "matrix.csv", "predictions.json")}})
        shutil.copytree(directory, source / "benchmark/completion_replication_initial")
        for name in ("LICENSE-RLVRAMBench", "instrumented_python_packages/verl/LICENSE",
                     "third_party/TransferQueue/LICENSE"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Synthetic notice\n")
    frozen = completion / "original/benchmark/completion_replication"
    freeze_hash = replication.sha256(frozen / "freeze.json")
    for source in (control, old_control):
        write_json(source / replication.CONTROL_DIRECTORY / replication.CONTROL_SEAL, {"test_only": True})
    control_seal = replication.sha256(control / replication.CONTROL_DIRECTORY / replication.CONTROL_SEAL)
    completion_amendment = completion / "completion-runtime-amendment-01.json"
    write_json(completion_amendment, {
        "original_freeze_sha256": freeze_hash,
        "initial_freeze_sha256": freeze_hash,
        "original_source_commit": "a" * 40,
        "amended_source_commit": "c" * 40,
        "original_source_root": "/former/completion-original",
        "amended_source_root": "/former/completion-amended",
    })
    control_amendment = control / replication.CONTROL_DIRECTORY / replication.CONTROL_AMENDMENT
    recorded_control = "/former/control/output/replication_control"
    write_json(control_amendment, {"base_seal_sha256": control_seal, "results_root": recorded_control})
    for panel in replication.PANELS:
        write_json(artifact / "output" / panel / "raw.json", {"peak_mib": 42})
        recorded = (replication.LIVE_PARENT / "rl/output/completion_replication"
                    if panel == "completion_replication" else recorded_control)
        write_outputs(panel, artifact / "benchmark" / panel / "results", 42, recorded)
    accounting = artifact / "benchmark/completion_replication/allocation_accounting.json"
    write_json(accounting, {"123": {"whole_allocation_gpu_seconds": 10}})
    # A source-local scientific loader, never a first-pair audit or model refit.
    (completion / "original/memory_tuner/completion_replication_protocol.py").write_text(f"""
import json
from pathlib import Path
def load_frozen(root, directory, anchor):
    assert root == Path(__file__).resolve().parents[1]
    assert not Path({str(artifact / 'output/completion_replication/raw.json')!r}).exists()
    assert not Path({str(artifact / 'output/replication_control/raw.json')!r}).exists()
    assert not Path({str(accounting)!r}).exists()
    assert not Path({str(completion_amendment)!r}).read_text()
    assert not Path({str(completion / replication.EXPORT_REPORT)!r}).read_text()
    assert not Path({str(completion / 'amended/memory_tuner/collect_completion_replication.py')!r}).exists()
    seal = json.loads((directory / 'freeze.json').read_text())
    return dict(freeze_sha256=anchor,
                protocol_sha256=seal['files_sha256']['protocol.json'],
                matrix_sha256=seal['files_sha256']['matrix.csv'],
                predictions=json.loads((directory / 'predictions.json').read_text()))
""")
    for panel, source in (("completion_replication", completion / "amended"),
                          ("replication_control", control)):
        (source / "memory_tuner" / f"collect_{panel}.py").write_text(collector_script(panel))
    amendment = json.loads(completion_amendment.read_text())
    for role in ("original", "amended"):
        amendment[role + "_runtime_sha256"] = {
            name: digest for name, digest in replication.inventory(completion / role).items()
            if name.endswith(".py")}
    write_json(completion_amendment, amendment)
    export_receipt(completion, completion_amendment)
    metadata = {
        "schema_version": 1,
        "code_sha256": {name: replication.sha256(ROOT / name)
                        for name in ("reproduce.py", "replication_reproduction.py")},
        "panels": {
            "completion_replication": {
                "source_tree_sha256": {cohort: replication.tree_sha256(completion / cohort)
                                       for cohort in ("original", "amended")},
                "freeze_sha256": freeze_hash,
                "runtime_amendment_sha256": replication.sha256(completion_amendment),
                "amendment_filename": completion_amendment.name,
                "allocation_accounting_sha256": replication.sha256(accounting),
            },
            "replication_control": {
                "source_tree_sha256": {"amended": replication.tree_sha256(control)},
                "execution_seal_sha256": control_seal,
                "runtime_amendment_sha256": replication.sha256(control_amendment),
            },
        },
    }
    manifest = artifact / "profiles/benchmark/publication-artifact-manifest.csv"
    tree = SimpleNamespace(
        work=work, artifact=artifact, manifest=manifest, completion=completion,
        control=control, metadata=metadata,
        release={"benchmark_protocol_version": "1.0", "replication_reproduction": metadata})
    rewrite_manifest(tree)
    return tree


def prepare(tree):
    return replication.prepare_panels(tree.artifact, getattr(tree, "code_root", ROOT),
                                      tree.release, tree.manifest, isolated=True)


def staged(tree):
    panels = prepare(tree)
    reference = tree.work / "reference-benchmark"
    (tree.artifact / "benchmark").rename(reference)
    replication.restore_panel_inputs(tree.artifact, reference, panels)
    return panels, reference


def fake_child(tree, calls):
    def run(arguments, work, hidden, *, cwd):
        assert work == tree.work and cwd.is_relative_to(tree.artifact / "reproduction_sources")
        assert replication.LIVE_PARENT in hidden
        assert tree.work / "reference-benchmark" in hidden
        calls.append((arguments, hidden, cwd))
        if arguments[0] == "-c":
            assert cwd == tree.completion / "original"
            assert tree.artifact / "output" in hidden
            assert tree.completion / "amended" in hidden
            directory = cwd / "benchmark/completion_replication"
            seal = replication.read_json(directory / "freeze.json")
            write_json(Path(arguments[-1]), dict(
                status="passed", freeze_sha256=arguments[-2],
                protocol_sha256=seal["files_sha256"]["protocol.json"],
                matrix_sha256=seal["files_sha256"]["matrix.csv"],
                predictions=replication.read_json(directory / "predictions.json"),
                fresh_refit_performed=False, prospective_outcomes_read=False))
            return
        panel = arguments[1].removeprefix("memory_tuner.collect_")
        assert cwd == (tree.completion / "amended" if panel == "completion_replication" else tree.control)
        assert tree.artifact / "output" / panel not in hidden
        output = Path(arguments[arguments.index("--output") + 1])
        write_outputs(panel, output, 42, tree.artifact / "output" / panel, seconds=.25)
    return run


def test_optional_panels_absent_preserve_old_workflow(tmp_path):
    assert replication.prepare_panels(tmp_path, ROOT, {}, tmp_path / "absent", isolated=False) == {}
    assert replication.reconstruct_panels(tmp_path, tmp_path, tmp_path, [], {}, None) == {}


def test_contract_digest_covers_full_canonical_contract_and_rejects_nan(panel_tree):
    contract = panel_tree.metadata
    expected = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert replication.contract_sha256(contract) == expected
    assert replication.contract_sha256(dict(reversed(list(contract.items())))) == expected
    changed = json.loads(json.dumps(contract))
    changed["code_sha256"]["reproduce.py"] = "0" * 64
    assert replication.contract_sha256(changed) != expected
    changed["new_field"] = float("nan")
    with pytest.raises(ValueError):
        replication.contract_sha256(changed)


def test_old_main_workflow_omits_replication_contract_digest(reconstruction_tree):
    reproduce.main()
    report = json.loads((reconstruction_tree.work / "verification.json").read_text())
    assert "replication_reproduction_contract_sha256" not in report


@pytest.mark.parametrize("mutation", [
    "missing_metadata", "missing_panel", "missing_original", "missing_reference", "missing_raw",
    "code_hash", "source_hash", "freeze", "amendment", "accounting", "unknown_field",
    "path_escape", "absolute_path", "linked_source", "linked_raw", "unmanifested", "changed_science",
])
def test_fail_closed_before_any_child(panel_tree, mutation):
    tree = panel_tree
    settings = tree.metadata["panels"]["completion_replication"]
    if mutation == "missing_metadata":
        tree.release.pop("replication_reproduction")
    elif mutation == "missing_panel":
        tree.metadata["panels"].pop("replication_control")
    elif mutation == "missing_original":
        (tree.completion / "original").rename(tree.work / "missing-original")
    elif mutation == "missing_reference":
        (tree.artifact / "benchmark/replication_control/results").rename(tree.work / "missing-reference")
    elif mutation == "missing_raw":
        (tree.artifact / "output/replication_control").rename(tree.work / "missing-raw")
    elif mutation == "code_hash":
        tree.metadata["code_sha256"]["reproduce.py"] = "0" * 64
    elif mutation == "source_hash":
        settings["source_tree_sha256"]["original"] = "0" * 64
    elif mutation in ("freeze", "amendment", "accounting"):
        key = {"freeze": "freeze_sha256", "amendment": "runtime_amendment_sha256",
               "accounting": "allocation_accounting_sha256"}[mutation]
        settings[key] = "0" * 64
    elif mutation == "unknown_field":
        settings["skip_verification"] = True
    elif mutation in ("path_escape", "absolute_path"):
        settings["amendment_filename"] = "../outside.json" if mutation == "path_escape" else "/outside.json"
    elif mutation in ("linked_source", "linked_raw"):
        path = tree.completion / "original" if mutation == "linked_source" else tree.artifact / "output/replication_control"
        outside = tree.work / "outside"
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=True)
    elif mutation == "unmanifested":
        (tree.control / "new_helper.py").write_text("# not manifest-bound\n")
        tree.metadata["panels"]["replication_control"]["source_tree_sha256"]["amended"] = replication.tree_sha256(tree.control)
    elif mutation == "changed_science":
        write_json(tree.completion / "original/benchmark/completion_replication/predictions.json",
                   [{"peak_mib": 43}])
    with pytest.raises((ValueError, KeyError)):
        prepare(tree)
    assert not (tree.work / "completion_replication-results").exists()


def test_new_panels_require_real_isolation(panel_tree):
    with pytest.raises(ValueError, match="isolate-analysis"):
        replication.prepare_panels(panel_tree.artifact, ROOT, panel_tree.release,
                                   panel_tree.manifest, isolated=False)


def test_two_cohort_dispatch_and_report_with_only_accounting_restored(panel_tree):
    tree = panel_tree
    panels, reference = staged(tree)
    assert {p.relative_to(tree.artifact / "benchmark").as_posix()
            for p in (tree.artifact / "benchmark").rglob("*") if p.is_file()} == {
                "completion_replication/allocation_accounting.json"}
    calls = []
    reports = replication.reconstruct_panels(
        tree.artifact, reference, tree.work, [reference], panels, fake_child(tree, calls))
    assert len(calls) == 3 and calls[0][0][0] == "-c"
    completion = reports["replication_reproduction"]["completion_replication"]
    assert completion["fresh_refit_performed"] is False
    assert completion["panel_files_matched"] == 4
    assert completion["collection_cpu_wall_seconds"] == {
        "published_seconds": 1., "reconstructed_seconds": .25}
    assert completion["allocation_accounting_sha256"]
    assert "--allocation-accounting" in calls[1][0]
    control = reports["replication_reproduction"]["replication_control"]
    assert control["panel_files_matched"] == 4
    assert "collection_cpu_wall_seconds" not in control
    assert "--allow-relocation" in calls[2][0]
    assert calls[2][0][calls[2][0].index("--base-source") + 1] == str(
        tree.control / "replication_control_base_source")


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), True, "1.0"])
@pytest.mark.parametrize("side", ["reference", "output"])
def test_only_positive_finite_completion_timing_can_be_excluded(tmp_path, value, side):
    for directory, seconds in ((tmp_path / "reference", 1.), (tmp_path / "output", .1)):
        write_outputs("completion_replication", directory, 42, "/same/root", seconds)
    path = tmp_path / side / "summary.json"
    record = json.loads(path.read_text())
    record[replication.TIMING_FIELD] = value
    write_json(path, record)
    with pytest.raises(ValueError, match="CPU timing"):
        replication.compare_outputs("completion_replication", tmp_path / "reference",
                                    tmp_path / "output", {})


@pytest.mark.parametrize("panel", replication.PANELS)
@pytest.mark.parametrize("mutation", ["science", "extra_field", "csv", "missing_file", "extra_file"])
def test_no_scientific_changes_or_file_set_changes_ignored(tmp_path, panel, mutation):
    for directory in (tmp_path / "reference", tmp_path / "output"):
        write_outputs(panel, directory, 42, "/same/root")
    output = tmp_path / "output"
    summary = output / ("summary.json" if panel == "completion_replication" else "replication_control_summary.json")
    if mutation in ("science", "extra_field"):
        record = json.loads(summary.read_text())
        record["peak_mib" if mutation == "science" else "new_field"] = 99
        write_json(summary, record)
    elif mutation == "csv":
        next(output.glob("*.csv")).write_text("condition,peak_mib\nbase,43\n")
    elif mutation == "missing_file":
        summary.unlink()
    else:
        (output / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError):
        replication.compare_outputs(panel, tmp_path / "reference", output, {})


@pytest.mark.parametrize("mutation", ["refit", "predictions", "anchor"])
def test_failed_prediction_replay_prevents_collection(panel_tree, mutation):
    tree = panel_tree
    panels, reference = staged(tree)
    calls = []
    original = fake_child(tree, calls)

    def bad_replay(arguments, *args, **kwargs):
        original(arguments, *args, **kwargs)
        assert arguments[0] == "-c", "Collector ran despite invalid prediction replay"
        path = Path(arguments[-1])
        result = json.loads(path.read_text())
        if mutation == "refit":
            result["fresh_refit_performed"] = True
        elif mutation == "predictions":
            result["predictions"][0]["peak_mib"] = 43
        else:
            result["freeze_sha256"] = "0" * 64
        write_json(path, result)

    with pytest.raises(ValueError, match="prediction replay"):
        replication.reconstruct_panels(tree.artifact, reference, tree.work, [reference], panels, bad_replay)
    assert len(calls) == 1 and not (tree.work / "completion_replication-results").exists()


@pytest.mark.parametrize("kind", ["outside", "symlink", "hidden", "missing"])
def test_run_child_rejects_unvalidated_cwd_before_subprocess(tmp_path, monkeypatch, kind):
    work, outside = tmp_path / "work", tmp_path / "outside"
    source = work / "artifact/source"
    source.mkdir(parents=True)
    outside.mkdir()
    hidden = []
    if kind == "outside":
        source = outside
    elif kind == "symlink":
        source = source / "linked"
        source.symlink_to(outside, target_is_directory=True)
    elif kind == "missing":
        source = source / "missing"
    else:
        hidden = [source]
    calls = []
    monkeypatch.setattr(reproduce.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="cwd"):
        reproduce.run_child(["-m", "memory_tuner.fake"], work, hidden, cwd=source)
    assert calls == []


def test_run_child_executes_source_cwd_and_strips_caller_fallback(tmp_path, monkeypatch):
    work = tmp_path / "work"
    source = work / "artifact/source"
    package = source / "memory_tuner"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    marker = work / "marker"
    (package / "fake.py").write_text(
        "from pathlib import Path\n"
        f"assert Path.cwd() == Path({str(source)!r})\n"
        f"Path({str(marker)!r}).write_text('archived')\n")
    fallback = tmp_path / "fallback"
    (fallback / "memory_tuner").mkdir(parents=True)
    (fallback / "memory_tuner/__init__.py").write_text("")
    (fallback / "memory_tuner/fake.py").write_text("raise AssertionError('live fallback')")
    monkeypatch.setenv("PYTHONPATH", str(fallback))
    reproduce.run_child(["-m", "memory_tuner.fake"], work, [], cwd=source)
    assert marker.read_text() == "archived"
    marker.unlink()
    (package / "fake.py").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        reproduce.run_child(["-m", "memory_tuner.fake"], work, [], cwd=source)
    assert not marker.exists()


def test_actual_bwrap_archived_cwd_and_prediction_target_hiding(panel_tree):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap not installed")
    available = subprocess.run(
        [bwrap, "--die-with-parent", "--ro-bind", "/", "/", "--", "/usr/bin/true"],
        capture_output=True, text=True, timeout=10, env={"PATH": os.defpath})
    if available.returncode:
        pytest.skip("bubblewrap namespace unavailable: " + available.stderr.strip())
    tree = panel_tree
    panels, reference = staged(tree)
    result = replication.reconstruct_panels(
        tree.artifact, reference, tree.work, [reference], panels, reproduce.run_child)
    assert result["replication_reproduction"]["completion_replication"]["prediction_replay_passed"]
    assert result["replication_reproduction"]["replication_control"]["panel_files_matched"] == 4
    assert (tree.artifact / "output/completion_replication/raw.json").is_file()
    assert (reference / "completion_replication/results/summary.json").is_file()


def test_extractor_preserves_empty_raw_directories_and_counts_them(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name in ("output/completion_replication/empty-slot",
                     "output/completion_replication/empty-slot/allocator-traces"):
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
        file = tarfile.TarInfo("raw.json")
        file.size = 2
        archive.addfile(file, io.BytesIO(b"{}"))
    payload.seek(0)
    decoder = SimpleNamespace(stdout=payload, wait=lambda: 0, poll=lambda: 0)
    monkeypatch.setattr(reproduce.subprocess, "Popen", lambda *args, **kwargs: decoder)
    destination = tmp_path / "extracted"
    assert reproduce.extract(tmp_path / "synthetic.tar.zst", destination) == 3
    empty = destination / "output/completion_replication/empty-slot/allocator-traces"
    assert empty.is_dir() and list(empty.iterdir()) == []
    assert (destination / "raw.json").read_bytes() == b"{}"


@pytest.mark.parametrize("field", ["archive_members", "manifest_entries"])
def test_main_rejects_incorrect_release_counts(reconstruction_tree, field):
    tree = reconstruction_tree
    tree.release[field] = 999
    write_json(tree.code / "release.json", tree.release)
    with pytest.raises(ValueError, match="count differs"):
        reproduce.main()
    assert not tree.children


@pytest.mark.parametrize("mode", ["normal", "omit_metadata", "disk_mutation", "memory_mutation"])
def test_main_integrates_new_panels_without_legacy_fallback(
        reconstruction_tree, panel_tree, monkeypatch, mode):
    tree, new = reconstruction_tree, panel_tree
    for name in ("reproduce.py", "replication_reproduction.py"):
        shutil.copy2(ROOT / name, tree.code / name)
    if mode != "omit_metadata":
        tree.release["replication_reproduction"] = new.metadata
    tree.release.update(archive_members=37, manifest_entries=1 + len(
        replication.manifest_entries(new.manifest)))
    write_json(tree.code / "release.json", tree.release)
    expected_contract = hashlib.sha256(json.dumps(
        new.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    old_extract, old_child = reproduce.extract, reproduce.run_child
    if mode == "memory_mutation":
        old_reconstruct = replication.reconstruct_panels
        old_prepare = replication.prepare_panels
        validated = []

        def retain_validated_contract(artifact, code, release, manifest, *, isolated):
            result = old_prepare(artifact, code, release, manifest, isolated=isolated)
            validated.append(release["replication_reproduction"])
            return result

        def mutate_after_collection(*args, **kwargs):
            result = old_reconstruct(*args, **kwargs)
            validated[0]["code_sha256"]["reproduce.py"] = "0" * 64
            return result

        monkeypatch.setattr(replication, "prepare_panels", retain_validated_contract)
        monkeypatch.setattr(replication, "reconstruct_panels", mutate_after_collection)

    def extract_with_panels(archive, destination):
        count = old_extract(archive, destination)
        for name in ("benchmark", "output", "reproduction_sources"):
            shutil.copytree(new.artifact / name, destination / name, dirs_exist_ok=True)
        manifest = destination / "profiles/benchmark/publication-artifact-manifest.csv"
        with manifest.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, ["path", "size_bytes", "sha256"])
            writer.writerows(replication.manifest_entries(new.manifest).values())
        return count

    calls = []
    adapted = SimpleNamespace(
        work=tree.work, artifact=tree.work / "artifact",
        completion=tree.work / "artifact/reproduction_sources/completion_replication",
        control=tree.work / "artifact/reproduction_sources/replication_control")
    new_child = fake_child(adapted, calls)

    def route(arguments, work, hidden, *, cwd=None):
        if cwd is not None:
            if mode == "disk_mutation":
                # Finalization may rewrite the release file while this run is
                # using its already-read contract. Do not certify that rewrite.
                changed = json.loads((tree.code / "release.json").read_text())
                changed["replication_reproduction"]["code_sha256"]["reproduce.py"] = "0" * 64
                write_json(tree.code / "release.json", changed)
            return new_child(arguments, work, hidden, cwd=cwd)
        assert work / "artifact/reproduction_sources" in hidden
        assert work / "artifact/output/completion_replication" in hidden
        return old_child(arguments, work, hidden)

    monkeypatch.setattr(reproduce, "extract", extract_with_panels)
    monkeypatch.setattr(reproduce, "run_child", route)
    if mode == "omit_metadata":
        with pytest.raises(ValueError, match="complete release metadata"):
            reproduce.main()
        assert not tree.children and not calls
    elif mode == "memory_mutation":
        with pytest.raises(ValueError, match="contract changed during reconstruction"):
            reproduce.main()
        assert not (tree.work / "verification.json").exists()
    else:
        reproduce.main()
        assert len(tree.children) == 7 and len(calls) == 3
        result = json.loads((tree.work / "verification.json").read_text())
        assert set(result["replication_reproduction"]) == set(replication.PANELS)
        assert result["archive_members"] == 37
        assert result["replication_reproduction_contract_sha256"] == expected_contract
        if mode == "disk_mutation":
            finalized = json.loads((tree.code / "release.json").read_text())
            assert replication.contract_sha256(finalized["replication_reproduction"]) != expected_contract


@pytest.mark.parametrize("mutation", ["raw_added", "accounting_changed", "source_changed"])
def test_inputs_rechecked_after_replay_before_collector(panel_tree, mutation):
    tree = panel_tree
    panels, reference = staged(tree)
    calls = []
    child = fake_child(tree, calls)

    def tamper(arguments, *args, **kwargs):
        child(arguments, *args, **kwargs)
        assert arguments[0] == "-c"
        if mutation == "raw_added":
            write_json(tree.artifact / "output/completion_replication/unmanifested.json", {})
        elif mutation == "accounting_changed":
            write_json(tree.artifact / "benchmark/completion_replication/allocation_accounting.json", {})
        else:
            (tree.completion / "original/unmanifested.py").write_text("# changed\n")

    with pytest.raises(ValueError, match="inventory changed|hash differs"):
        replication.reconstruct_panels(tree.artifact, reference, tree.work, [reference], panels, tamper)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", [
    "missing", "extra_entry", "anchor", "status", "count", "size", "runtime_hash",
    "cpu_count", "cpu_anchor", "notice", "receipt_escape",
])
def test_completion_export_receipt_is_strict_and_manifest_bound(panel_tree, mutation):
    tree = panel_tree
    path = tree.completion / replication.EXPORT_REPORT
    report = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra_entry":
        write_json(tree.completion / "unexpected.json", {})
    elif mutation == "receipt_escape":
        moved = tree.work / "outside-receipt.json"
        path.rename(moved)
        path.symlink_to(moved)
    else:
        if mutation == "anchor":
            report["amendment_sha256"] = "0" * 64
        elif mutation == "status":
            report["status"] = "not_validated"
        elif mutation == "count":
            report["trees"]["original"]["file_count"] += 1
        elif mutation == "size":
            report["trees"]["original"]["files"]["LICENSE-RLVRAMBench"]["size_bytes"] += 1
        elif mutation == "runtime_hash":
            report["trees"]["original"]["files"]["memory_tuner/__init__.py"]["sha256"] = "0" * 64
        elif mutation == "cpu_count":
            report["cpu_validation"]["planned_slots"] += 1
        elif mutation == "cpu_anchor":
            report["cpu_validation"]["freeze_sha256"] = "0" * 64
        elif mutation == "notice":
            report["trees"]["original"]["notices"].pop("LICENSE-RLVRAMBench")
        write_json(path, report)
    # Even an internally authenticated archive must have a self-consistent receipt.
    rewrite_manifest(tree)
    with pytest.raises((ValueError, KeyError)):
        prepare(tree)


@pytest.fixture
def accounting_tree(panel_tree, tmp_path):
    tree = panel_tree
    helper = Path(os.environ.get(
        "RLVRAM_ACCOUNTING_HELPER_SOURCE", str(ROOT / "memory_tuner/replication_control_accounting.py")))
    if not helper.is_file():
        pytest.skip("Parent accounting helper not staged; set RLVRAM_ACCOUNTING_HELPER_SOURCE to its read-only path")
    tree.code_root = tmp_path / "accounting-code"
    tree.code_root.mkdir()
    for name in ("reproduce.py", "replication_reproduction.py"):
        shutil.copy2(ROOT / name, tree.code_root / name)
    for name in replication.ACCOUNTING_CODE_FILES:
        source = helper if name.endswith("/replication_control_accounting.py") else ROOT / name
        for root in (tree.code_root, tree.artifact):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    from memory_tuner.capture_allocation_accounting import capture, timestamp
    block = tree.artifact / "output/replication_control/blocks/control-a"
    start = timestamp("2026-09-16T10:00:00")
    write_json(block / "replication_control_block.json", {
        "job_id": "123", "block_id": "control-a",
        "allocation": {"started_at_epoch": start, "hostname": "node.example"}})
    (block / "replication_control_ledger.jsonl").write_text(
        json.dumps(dict(event="claimed", job_id="123", block_id="control-a",
                        timestamp_ns=int((start + 1) * 1e9))) + "\n" +
        json.dumps(dict(event="terminal", timestamp_ns=int((start + 60.5) * 1e9))) + "\n")
    rows = "\n".join(
        f"123_0{suffix}|123{suffix}|{name}|FAILED|2:0|"
        "2026-09-16T10:00:00|2026-09-16T10:01:00|60|cpu=32,gres/gpu=2|node|NonZeroExitCode"
        for suffix, name in (("", "control"), (".batch", "batch")))
    records = capture(rows, {"123": dict(
        pair_id="control-a", pair_manifest_sha256=replication.sha256(block / "replication_control_block.json"),
        pair_ledger_sha256=replication.sha256(block / "replication_control_ledger.jsonl"))})
    accounting = tree.artifact / "benchmark/replication_control/allocation_accounting.json"
    write_json(accounting, records)
    tree.metadata["panels"]["replication_control"].update(
        allocation_accounting_sha256=replication.sha256(accounting),
        accounting_code_sha256={name: replication.sha256(tree.code_root / name)
                                for name in replication.ACCOUNTING_CODE_FILES})
    tree.accounting = accounting
    rewrite_manifest(tree)
    return tree


@pytest.mark.parametrize("mutation", [
    "omitted_fields", "half_fields", "missing_input", "input_hash", "missing_code",
    "incomplete_closure", "code_hash", "different_archive_code", "unmanifested_code",
])
def test_optional_accounting_requires_complete_input_and_code_anchors(accounting_tree, mutation):
    tree = accounting_tree
    settings = tree.metadata["panels"]["replication_control"]
    name = "memory_tuner/replication_control_accounting.py"
    if mutation == "omitted_fields":
        settings.pop("allocation_accounting_sha256")
        settings.pop("accounting_code_sha256")
    elif mutation == "half_fields":
        settings.pop("accounting_code_sha256")
    elif mutation == "missing_input":
        tree.accounting.unlink()
    elif mutation == "input_hash":
        settings["allocation_accounting_sha256"] = "0" * 64
    elif mutation == "missing_code":
        (tree.code_root / name).unlink()
    elif mutation == "incomplete_closure":
        settings["accounting_code_sha256"].pop("memory_tuner/core.py")
    elif mutation == "code_hash":
        settings["accounting_code_sha256"][name] = "0" * 64
    elif mutation == "different_archive_code":
        (tree.artifact / name).write_text("# different from pinned public helper\n")
    else:
        rows = replication.manifest_entries(tree.manifest)
        rows.pop(name)
        with tree.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, ["path", "size_bytes", "sha256"])
            writer.writeheader()
            writer.writerows(rows.values())
    with pytest.raises(ValueError):
        prepare(tree)


@pytest.mark.parametrize("invalid_records", [False, True])
def test_real_accounting_helper_in_namespace_no_batch_double_count(accounting_tree, invalid_records):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap not installed")
    available = subprocess.run(
        [bwrap, "--die-with-parent", "--ro-bind", "/", "/", "--", "/usr/bin/true"],
        capture_output=True, text=True, timeout=10, env={"PATH": os.defpath})
    if available.returncode:
        pytest.skip("bubblewrap namespace unavailable: " + available.stderr.strip())
    tree = accounting_tree
    if invalid_records:
        records = json.loads(tree.accounting.read_text())
        records.append(records[0])
        write_json(tree.accounting, records)
        tree.metadata["panels"]["replication_control"]["allocation_accounting_sha256"] = replication.sha256(tree.accounting)
        rewrite_manifest(tree)
    panels, reference = staged(tree)
    calls = []
    synthetic = fake_child(tree, calls)
    cost_calls = []

    def route(arguments, work, hidden, *, cwd):
        if arguments[:2] == ["-m", "memory_tuner.replication_control_accounting"]:
            assert cwd == tree.artifact and reference in hidden and replication.LIVE_PARENT in hidden
            cost_calls.append(arguments)
            return reproduce.run_child(arguments, work, hidden, cwd=cwd)
        return synthetic(arguments, work, hidden, cwd=cwd)

    if invalid_records:
        with pytest.raises(subprocess.CalledProcessError):
            replication.reconstruct_panels(tree.artifact, reference, tree.work, [reference], panels, route)
        assert not (tree.work / "replication_control-accounting.json").exists()
    else:
        result = replication.reconstruct_panels(tree.artifact, reference, tree.work, [reference], panels, route)
        control = result["replication_reproduction"]["replication_control"]
        assert control["whole_allocation_accounting_reconstruction_passed"]
        cost = control["whole_allocation_accounting"]
        assert cost["allocation_count"] == 1 and cost["whole_allocation_gpu_seconds"] == 120
        assert cost["allocations"][0]["batch_step_gpu_seconds"] == 120
        assert cost["allocations"][0]["scheduler_state"] == "FAILED"
    assert len(cost_calls) == 1
    assert set(replication.inventory(tree.work / "replication_control-results")) == replication.OUTPUTS["replication_control"]
