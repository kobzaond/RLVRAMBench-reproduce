import io
import tarfile
from pathlib import Path

import pytest

from reproduce import check_member, normalize_paths, sha256


@pytest.mark.parametrize("name", ["/outside", "../outside", "a/../../outside"])
def test_member_rejects_path_escape(tmp_path, name):
    with pytest.raises(ValueError):
        check_member(tarfile.TarInfo(name), tmp_path)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize("target", ["/outside", "../../outside"])
def test_member_rejects_escaping_links(tmp_path, kind, target):
    member = tarfile.TarInfo("inside/link")
    member.type = kind
    member.linkname = target
    with pytest.raises(ValueError):
        check_member(member, tmp_path)


def test_member_accepts_relative_internal_link(tmp_path):
    member = tarfile.TarInfo("inside/link")
    member.type = tarfile.SYMTYPE
    member.linkname = "../trial.json"
    check_member(member, tmp_path)


def test_member_rejects_device(tmp_path):
    member = tarfile.TarInfo("device")
    member.type = tarfile.CHRTYPE
    with pytest.raises(ValueError):
        check_member(member, tmp_path)


def test_normalization_changes_paths_not_measurements():
    assert normalize_paths(
        "/old/artifact/output/run 3980 -31 588", [Path("/old/artifact")]
    ) == "<ARTIFACT_ROOT>/output/run 3980 -31 588"


def test_digest(tmp_path):
    path = tmp_path / "small"
    path.write_bytes(b"abc")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# Integration regressions use only synthetic evidence in pytest's temporary tree.
import csv
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import reproduce
from memory_tuner import artifact_paths


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def reconstruction_tree(tmp_path, monkeypatch):
    code, original, clone, other_clone = (
        tmp_path / name for name in ("code", "original", "clone", "other-clone"))
    for directory in (code / "benchmark", original, clone, other_clone):
        directory.mkdir(parents=True)
        (directory / "reference.csv").write_text("reference outcome\n")
    archive = tmp_path / "synthetic.tar.zst"
    archive.write_bytes(b"synthetic archive: extraction is stubbed")
    work = tmp_path / "work"
    original_summary = {
        "scope": {"principal_fresh_processes": 588,
                  "supporting_historical_source_processes": 16}}
    control_summary = {"completed": 24, "attempts": 26}
    decision_summary = {
        "planned_slots": 48, "recorded_attempts": 48,
        "validated_memory_outcomes": 48}
    release = {
        "archive": {"sha256": sha256(archive)},
        "benchmark_protocol_version": "1.0",
        "expected_derived_files": 1, "expected_review_derived_files": 1,
        "expected_decision_derived_files": 1,
        "review_control_completed_processes": 24,
        "review_control_attempted_processes": 26,
        "decision_planned_slots": 48, "decision_recorded_attempts": 48,
        "decision_validated_memory_outcomes": 48,
    }
    _write_json(code / "release.json", release)
    extracted, children = [], []

    def extract_synthetic(path, destination):
        assert path == archive
        extracted.append(destination)
        profiles = destination / "profiles"
        (profiles / "strengthening/same_node").mkdir(parents=True)
        _write_json(profiles / "standard_grpo/summary.json", original_summary)
        _write_json(profiles / "review_revision/control-summary.json", control_summary)
        raw = destination / "output/raw.txt"
        raw.parent.mkdir()
        raw.write_text("raw evidence\n")
        manifest = profiles / "benchmark/publication-artifact-manifest.csv"
        manifest.parent.mkdir()
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, ["path", "size_bytes", "sha256"], lineterminator="\n")
            writer.writeheader()
            writer.writerow({"path": "output/raw.txt", "size_bytes": raw.stat().st_size,
                             "sha256": sha256(raw)})
        benchmark = destination / "benchmark"
        benchmark.mkdir()
        (benchmark / "outcomes.csv").write_text("configuration_id,observed_state\n")
        for name in ("protocol.json", "matrix.csv", "amendment-01.json",
                     "submission.json", "repairs.json"):
            path = benchmark / "decision" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        _write_json(benchmark / "decision/results/collection_summary.json", decision_summary)
        return 37

    def analysis_stub(arguments, passed_work, hidden):
        assert passed_work == work
        children.append((list(arguments), list(hidden)))
        artifact = work / "artifact"
        if arguments[:2] == ["-m", "memory_tuner.standard_grpo_paper_analysis"]:
            # References must be moved, with only frozen decision inputs restored.
            assert not (artifact / "profiles/standard_grpo").exists()
            assert not (artifact / "benchmark/outcomes.csv").exists()
            inputs = artifact / "benchmark/decision"
            assert {p.name for p in inputs.iterdir()} == {
                "protocol.json", "matrix.csv", "amendment-01.json",
                "submission.json", "repairs.json"}
            _write_json(work / "results/summary.json", original_summary)
        elif arguments[:2] == ["-m", "memory_tuner.review_evidence_analysis"]:
            _write_json(work / "review-results/control-summary.json", control_summary)
        elif arguments[:2] == ["-m", "memory_tuner.make_review_figures"]:
            figures = work / "figures"
            figures.mkdir()
            for name in ("standard_grpo_boundary", "standard_grpo_mechanisms",
                         "standard_grpo_temporal", "review_instrumentation_batch"):
                for suffix in (".pdf", ".png"):
                    (figures / (name + suffix)).write_bytes(b"synthetic figure")
        elif arguments[0] == str(code / "export_benchmark.py"):
            (work / "benchmark").mkdir()
            (work / "benchmark/outcomes.csv").write_text("configuration_id,observed_state\n")
        elif arguments[:2] == ["-m", "memory_tuner.collect_decision_benchmark"]:
            _write_json(work / "decision-results/collection_summary.json", decision_summary)
        else:
            assert arguments[1] in (
                "memory_tuner.review_attempt_workload_audit",
                "memory_tuner.review_control_analysis")

    monkeypatch.setattr(reproduce, "ROOT", code)
    monkeypatch.setattr(artifact_paths, "ORIGINAL_ROOT", original)
    monkeypatch.setattr(reproduce, "extract", extract_synthetic)
    monkeypatch.setattr(reproduce, "run_child", analysis_stub)
    argv = ["reproduce.py", "--archive", str(archive), "--work-dir", str(work),
            "--isolate-analysis"]
    monkeypatch.setattr(sys, "argv", argv)
    return SimpleNamespace(
        code=code, original=original, clone=clone, other_clone=other_clone,
        archive=archive, work=work, release=release, extracted=extracted,
        children=children, argv=argv)


def test_main_restores_only_inputs_hides_code_and_clones_and_records_evidence(reconstruction_tree):
    tree = reconstruction_tree
    tree.argv.extend(["--hide-path", str(tree.clone),
                      "--hide-path", str(tree.other_clone),
                      "--hide-path", str(tree.clone)])
    reproduce.main()
    expected = {
        tree.work / "reference-profiles", tree.work / "reference-benchmark",
        tree.original, tree.code / "benchmark", tree.clone, tree.other_clone}
    assert len(tree.children) == 7
    for _, hidden in tree.children:
        assert set(hidden) == expected and len(hidden) == len(expected)
    report = json.loads((tree.work / "verification.json").read_text())
    manifest = tree.work / "reference-profiles/benchmark/publication-artifact-manifest.csv"
    assert report["evidence_manifest_sha256"] == sha256(manifest)
    assert report["manifest_entries_verified"] == 1
    assert report["archive_members"] == 37
    assert report["decision_validated_memory_outcomes"] == 48
    assert report["curated_benchmark_files_matched"] == 1
    assert report["decision_panel_files_matched"] == 1
    assert report["child_namespace_reference_inaccessibility_checked"]
    assert set(report["hidden_paths"]) == {str(p) for p in expected}
    assert not report["network_namespace_isolation_claimed"]


@pytest.mark.parametrize("case", [
    "without_isolation", "missing_directory", "file", "work_in_clone",
    "code_in_hidden_directory", "work_in_code_benchmark", "symlink_to_clone",
])
def test_main_rejects_invalid_hiding_before_extraction(reconstruction_tree, tmp_path, case):
    tree = reconstruction_tree
    hidden = tree.clone
    if case == "without_isolation":
        tree.argv.remove("--isolate-analysis")
    elif case == "missing_directory":
        hidden = tmp_path / "missing"
    elif case == "file":
        hidden = tree.archive
    elif case == "work_in_clone":
        tree.argv[tree.argv.index("--work-dir") + 1] = str(tree.clone / "work")
    elif case == "code_in_hidden_directory":
        hidden = tree.code
    elif case == "work_in_code_benchmark":
        tree.argv[tree.argv.index("--work-dir") + 1] = str(tree.code / "benchmark/work")
    elif case == "symlink_to_clone":
        alias = tmp_path / "alias"
        alias.symlink_to(tree.clone, target_is_directory=True)
        hidden = alias
        tree.argv[tree.argv.index("--work-dir") + 1] = str(alias / "work")
    tree.argv.extend(["--hide-path", str(hidden)])
    with pytest.raises(ValueError):
        reproduce.main()
    assert not tree.extracted and not tree.children
    assert not tree.work.exists()


def test_main_rejects_wrong_validated_outcome_count(reconstruction_tree):
    tree = reconstruction_tree
    tree.release["decision_validated_memory_outcomes"] = 47
    _write_json(tree.code / "release.json", tree.release)
    with pytest.raises(ValueError, match="slot/attempt accounting"):
        reproduce.main()
    assert not (tree.work / "verification.json").exists()


def test_run_child_probes_identical_mounts_before_every_analysis(tmp_path, monkeypatch):
    hidden = [tmp_path / "code/benchmark", tmp_path / "clone"]
    for path in hidden:
        path.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    calls = []
    monkeypatch.setattr(reproduce.os, "environ",
                        {"PATH": os.defpath, "PYTHONPATH": str(tmp_path / "reference-code")})
    monkeypatch.setattr(reproduce.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(reproduce.subprocess, "run",
                        lambda command, **kwargs: calls.append((command, kwargs)))
    for script in ("first.py", "second.py"):
        reproduce.run_child([script], work, hidden)
    assert len(calls) == 4
    for probe, analysis in (calls[:2], calls[2:]):
        probe_command, probe_options = probe
        command, options = analysis
        assert probe_command[:probe_command.index("--")] == command[:command.index("--")]
        assert json.loads(probe_command[-1]) == [str(p) for p in hidden]
        assert "-c" in probe_command
        assert [command[i + 1] for i, v in enumerate(command[:-1]) if v == "--tmpfs"] == [
            "/run", *map(str, hidden)]
        assert probe_options["check"] and options["check"]
        assert "PYTHONPATH" not in options["env"]


@pytest.mark.parametrize("isolated", [False, True])
def test_run_child_overrides_caller_tmpdir_without_changing_parent_environment(
        tmp_path, monkeypatch, isolated):
    work = tmp_path / "work"
    work.mkdir()
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    caller_tmp = tmp_path / "caller-tmp"
    caller_tmp.mkdir()
    parent = {**os.environ, "TMPDIR": str(caller_tmp)}
    monkeypatch.setattr(reproduce.os, "environ", parent)
    monkeypatch.setattr(reproduce.shutil, "which", lambda name: "/usr/bin/bwrap")
    calls = []
    monkeypatch.setattr(reproduce.subprocess, "run",
                        lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.chdir(tmp_path)
    # Resolve relative work paths before passing TMPDIR to a differently located child.
    for script in ("first.py", "second.py"):
        reproduce.run_child([script], Path("work"), [hidden] if isolated else [])
    assert len(calls) == (4 if isolated else 2)
    assert (work / "tmp").is_dir() and not (work / "tmp").is_symlink()
    for _, kwargs in calls:
        assert kwargs["env"]["TMPDIR"] == str(work / "tmp")
        for name in ("HOME", "TMP", "TEMP"):
            assert kwargs["env"].get(name) == parent.get(name)
    assert os.environ["TMPDIR"] == str(caller_tmp)
    assert not any(caller_tmp.iterdir())


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("kind", ["external_directory", "dangling_external", "internal_directory"])
def test_run_child_rejects_symlinked_tmp_before_any_subprocess(
        tmp_path, monkeypatch, isolated, kind):
    work = tmp_path / "work"
    work.mkdir()
    target = work / "other" if kind == "internal_directory" else tmp_path / "outside"
    if kind != "dangling_external":
        target.mkdir()
        (target / "untouched").write_text("preserve\n")
    (work / "tmp").symlink_to(target, target_is_directory=True)
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    calls = []
    monkeypatch.setattr(reproduce.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="temporary directory"):
        reproduce.run_child(["must-not-run.py"], work, [hidden] if isolated else [])
    assert not calls and (work / "tmp").is_symlink()
    if kind == "dangling_external":
        assert not target.exists()
    else:
        assert list(target.iterdir()) == [target / "untouched"]
        assert (target / "untouched").read_text() == "preserve\n"


def test_failed_namespace_probe_prevents_analysis_launch(tmp_path, monkeypatch):
    hidden = tmp_path / "visible-reference"
    hidden.mkdir()
    (hidden / "outcomes.csv").write_text("precomputed outcomes\n")
    work = tmp_path / "work"
    work.mkdir()
    actual_run = subprocess.run
    calls = []

    def simulate_failed_hiding(command, **kwargs):
        calls.append(command)
        assert "-c" in command, "Analysis ran after its reference-visibility probe failed"
        index = command.index("-c")
        # Execute the real probe in a namespace where hiding failed.
        return actual_run([sys.executable, *command[index:]], check=True,
                          capture_output=True, env={"PATH": os.defpath})

    monkeypatch.setattr(reproduce.os, "environ", {"PATH": os.defpath})
    monkeypatch.setattr(reproduce.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(reproduce.subprocess, "run", simulate_failed_hiding)
    with pytest.raises(subprocess.CalledProcessError) as error:
        reproduce.run_child(["must-not-run.py"], work, [hidden])
    assert b"Reference evidence remains visible" in error.value.stderr
    assert len(calls) == 1


@pytest.mark.parametrize("isolated", [False, True])
def test_run_child_creates_real_tempfile_and_hides_references_when_isolated(
        tmp_path, monkeypatch, isolated):
    if isolated:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            pytest.skip("bubblewrap is not installed")
        preflight = subprocess.run(
            [bwrap, "--die-with-parent", "--ro-bind", "/", "/", "--", "/usr/bin/true"],
            capture_output=True, text=True, timeout=10, env={"PATH": os.defpath})
        if preflight.returncode:
            pytest.skip("bubblewrap namespace unavailable: " + preflight.stderr.strip())
    code, clone, work = (tmp_path / name for name in ("code", "clone", "work"))
    hidden = [code / "benchmark", clone]
    for path in hidden:
        path.mkdir(parents=True)
        (path / "outcomes.csv").write_text("hidden reference\n")
    work.mkdir()
    marker = work / "analysis-ran"
    script = f"""
import os, tempfile
from pathlib import Path
paths = {json.dumps([str(p) for p in hidden])}
assert all((not any(Path(p).iterdir())) is {isolated!r} for p in paths)
assert os.environ["TMPDIR"] == {str(work / "tmp")!r}
with tempfile.NamedTemporaryFile() as handle:
    assert Path(handle.name).parent == Path(os.environ["TMPDIR"])
    handle.write(b"writable temporary file")
    handle.flush()
    assert Path(handle.name).read_bytes() == b"writable temporary file"
Path({str(marker)!r}).write_text("passed")
"""
    monkeypatch.setattr(reproduce, "ROOT", code)
    monkeypatch.setattr(reproduce.os, "environ", {"PATH": os.defpath})
    reproduce.run_child(["-c", script], work, hidden if isolated else [])
    assert marker.read_text() == "passed"
    assert all((path / "outcomes.csv").is_file() for path in hidden)


def _estimation_outputs(directory, counts, release, artifact_root):
    summary = {key: counts[key] for key in reproduce.ESTIMATION_COUNT_FIELDS}
    summary["pair_errors"] = {"pair-0": f"Missing {artifact_root}/output/prospective_estimation/trial.json"}
    values = {
        "summary": summary,
        "evaluation": {"unresolved_configurations": counts["unresolved_configurations"]},
        "provenance": {
            "execution_commits_expected": release["estimation_execution_commits"],
            "prediction_seal_sha256": release["estimation_prediction_seal_sha256"],
            "protocol_sha256": release["estimation_protocol_sha256"],
            "execution_commits_file_sha256": release["estimation_execution_commits_sha256"],
            "allocation_accounting_sha256": release.get("estimation_allocation_accounting_sha256"),
        },
        "processes": [{"experiment_id": "partial", "state": "unresolved"}],
        "configurations": [], "configurations_flat": [], "pairs": [], "costs": {},
    }
    for name, value in values.items():
        _write_json(directory / (name + ".json"), value)
    for name in ("processes", "configurations", "configurations_flat", "pairs"):
        (directory / (name + ".csv")).write_text("configuration_id,state\npartial,unresolved\n")
    _write_json(directory / "output_manifest.json", {
        path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in directory.iterdir() if path.is_file() and path.name != "output_manifest.json"
    })


@pytest.fixture
def estimation_reconstruction_tree(reconstruction_tree, monkeypatch):
    tree = reconstruction_tree
    counts = {
        "planned_processes": 36, "planned_pairs": 18, "planned_configurations": 12,
        "eligible_processes": 29, "completed_processes": 22, "unresolved_processes": 7,
        "known_launched_processes": 34, "unknown_launch_status_processes": 1,
        "resolved_configurations": 8, "unresolved_configurations": 4,
    }
    tree.release.update({
        "source_commit": "c" * 40, "expected_estimation_derived_files": 13,
        "estimation_counts": counts,
        "estimation_execution_commits": {
            f"pair-{index}": ("a" if index < 2 else "b") * 40 for index in range(18)},
    })
    source_records = {}
    for name in reproduce.ESTIMATION_REQUIRED_SOURCE_FILES:
        path = tree.code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# authenticated synthetic source: " + name)
        source_records[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    _write_json(tree.code / "source-provenance.json", {
        "source_commit": tree.release["source_commit"], "files": source_records,
    })
    frozen = tree.code / "benchmark/estimation"
    for name in reproduce.ESTIMATION_INPUT_FILES:
        _write_json(frozen / name, {})
    _write_json(frozen / "execution_commits.json", tree.release["estimation_execution_commits"])
    _write_json(frozen / "allocation_accounting.json", [])
    for name, field in (
        ("protocol.json", "estimation_protocol_sha256"),
        ("prediction_freeze.json", "estimation_prediction_seal_sha256"),
        ("execution_commits.json", "estimation_execution_commits_sha256"),
        ("allocation_accounting.json", "estimation_allocation_accounting_sha256"),
    ):
        tree.release[field] = sha256(frozen / name)
    _estimation_outputs(frozen / "results", counts, tree.release, tree.original)
    _write_json(frozen / "unpermitted-target-outcomes.json", {"must_not_restore": True})
    _write_json(tree.code / "release.json", tree.release)
    original_extract = reproduce.extract
    original_child = reproduce.run_child
    tree.estimation_calls = []
    tree.estimation_bad_verification = {}
    tree.estimation_counts = dict(counts)
    tree.estimation_corrupt_output = False

    def extract_synthetic(archive, destination):
        count = original_extract(archive, destination)
        shutil.copytree(frozen, destination / "benchmark/estimation")
        for name in source_records:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tree.code / name, path)
        for name in ("prospective_estimation", "estimation-device-preflight"):
            path = destination / "output" / name / "raw.json"
            _write_json(path, {"raw_target_outcome": "not a fitting input"})
        (destination / "logs").mkdir()
        (destination / "logs/matched-gpu-123.out").write_text("target scheduler evidence")
        (destination / "paper").mkdir()
        (destination / "paper/manuscript.tex").write_text("Published target outcome table")
        return count

    def analysis_stub(arguments, work, hidden):
        artifact = work / "artifact"
        directory = artifact / "benchmark/estimation"
        assert not (directory / "results").exists()
        assert not (directory / "unpermitted-target-outcomes.json").exists()
        if arguments[:2] == ["-m", "memory_tuner.verify_estimation"]:
            tree.estimation_calls.append((list(arguments), list(hidden)))
            assert "--refit" in arguments
            assert Path(arguments[arguments.index("--root") + 1]) == artifact
            assert (artifact / "benchmark/outcomes.csv").read_bytes() == (
                work / "benchmark/outcomes.csv").read_bytes()
            assert set(path.name for path in directory.iterdir()) == (
                set(reproduce.ESTIMATION_INPUT_FILES)
                | ({"allocation_accounting.json"} if tree.release.get(
                    "estimation_allocation_accounting_sha256") is not None else set()))
            verification = {
                "status": "passed", "fresh_refit_compared": True,
                "prospective_outcomes_read": False,
                "prediction_seal_sha256": tree.release["estimation_prediction_seal_sha256"],
                "protocol_sha256": tree.release["estimation_protocol_sha256"],
                **tree.estimation_bad_verification,
            }
            _write_json(Path(arguments[arguments.index("--output") + 1]), verification)
        elif arguments[:2] == ["-m", "memory_tuner.collect_estimation_study"]:
            assert len(tree.estimation_calls) == 1, "Collector ran before independent replay/refit"
            tree.estimation_calls.append((list(arguments), list(hidden)))
            for option, expected in (
                ("--root", artifact), ("--source-root", artifact),
                ("--protocol-dir", directory),
                ("--execution-commits", directory / "execution_commits.json"),
            ):
                assert Path(arguments[arguments.index(option) + 1]) == expected
            if tree.release.get("estimation_allocation_accounting_sha256") is not None:
                assert Path(arguments[arguments.index("--allocation-accounting") + 1]) == (
                    directory / "allocation_accounting.json")
            else:
                assert "--allocation-accounting" not in arguments
            assert arguments[arguments.index("--protocol-sha256") + 1] == (
                tree.release["estimation_protocol_sha256"])
            assert arguments[arguments.index("--prediction-seal-sha256") + 1] == (
                tree.release["estimation_prediction_seal_sha256"])
            output = Path(arguments[arguments.index("--output") + 1])
            _estimation_outputs(output, tree.estimation_counts, tree.release, artifact)
            if tree.estimation_corrupt_output:
                (output / "processes.csv").write_text("silently corrupted")
        else:
            original_child(arguments, work, hidden)

    monkeypatch.setattr(reproduce, "extract", extract_synthetic)
    monkeypatch.setattr(reproduce, "run_child", analysis_stub)
    return tree


def test_estimation_restores_only_frozen_inputs_and_replays_before_raw_collection(
        estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    reproduce.main()
    assert len(tree.estimation_calls) == 2
    replay, collector = tree.estimation_calls
    common = set(tree.children[0][1])
    artifact = tree.work / "artifact"
    assert set(collector[1]) == common
    assert set(replay[1]) == common | {
        artifact / "output/prospective_estimation",
        artifact / "output/estimation_host_capacity",
        artifact / "output/estimation-device-preflight", artifact / "logs",
        artifact / "paper",
        artifact / "benchmark/estimation/allocation_accounting.json",
        artifact / "benchmark/host_capacity/allocation_accounting.json",
        tree.work / "estimation-results",
    }
    result = json.loads((tree.work / "verification.json").read_text())
    assert result["estimation_counts"]["eligible_processes"] == 29
    assert result["estimation_counts"]["resolved_configurations"] == 8
    assert result["estimation_panel_files_matched"] == 13
    assert result["estimation_execution_commits"] == tree.release["estimation_execution_commits"]
    assert result["estimation_source_files_verified"] == len(reproduce.ESTIMATION_REQUIRED_SOURCE_FILES)
    assert (tree.work / "estimation-results/output_manifest.json").read_bytes() != (
        tree.work / "reference-benchmark/estimation/results/output_manifest.json").read_bytes()
    for field in (
        "estimation_independent_raw_reconstruction_passed",
        "estimation_reference_outcomes_hidden_during_analysis",
        "estimation_prediction_replay_passed", "estimation_prediction_replay_outcomes_hidden",
        "estimation_fresh_refit_compared",
    ):
        assert result[field] is True


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("fresh_refit_compared", False),
    ("prospective_outcomes_read", True),
    ("prediction_seal_sha256", "0" * 64), ("protocol_sha256", "0" * 64),
])
def test_estimation_replay_failure_prevents_collector_and_success_report(
        estimation_reconstruction_tree, field, value):
    tree = estimation_reconstruction_tree
    tree.estimation_bad_verification[field] = value
    with pytest.raises(ValueError, match="replay/refit"):
        reproduce.main()
    assert len(tree.estimation_calls) == 1
    assert not (tree.work / "verification.json").exists()
    assert not (tree.work / "estimation-results").exists()


def test_estimation_changed_source_fails_before_any_analysis(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    (tree.code / "memory_tuner/estimation_baselines.py").write_text("# unpinned change")
    with pytest.raises(ValueError, match="analysis source differs"):
        reproduce.main()
    assert not tree.children and not tree.estimation_calls


def test_estimation_missing_source_closure_fails_before_any_analysis(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    path = tree.code / "source-provenance.json"
    provenance = json.loads(path.read_text())
    provenance["files"].pop("memory_tuner/device_contract.py")
    _write_json(path, provenance)
    with pytest.raises(ValueError, match="analysis closure"):
        reproduce.main()
    assert not tree.children and not tree.estimation_calls


def test_estimation_cannot_be_disabled_by_dropping_release_metadata(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    tree.release.pop("expected_estimation_derived_files")
    _write_json(tree.code / "release.json", tree.release)
    with pytest.raises(ValueError, match="collector references disagree"):
        reproduce.main()
    assert not tree.children and not tree.estimation_calls


def test_estimation_wrong_count_prevents_success_report(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    tree.estimation_counts["eligible_processes"] = 36
    with pytest.raises(ValueError, match="Estimation reconstruction differs"):
        reproduce.main()
    assert not (tree.work / "verification.json").exists()


def test_estimation_output_digest_corruption_prevents_success_report(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    tree.estimation_corrupt_output = True
    with pytest.raises(ValueError, match="output manifest mismatch"):
        reproduce.main()
    assert not (tree.work / "verification.json").exists()


def test_estimation_without_isolation_does_not_claim_hidden_outcomes(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    tree.argv.remove("--isolate-analysis")
    reproduce.main()
    result = json.loads((tree.work / "verification.json").read_text())
    assert result["estimation_independent_raw_reconstruction_passed"]
    assert result["estimation_prediction_replay_passed"]
    assert not result["estimation_reference_outcomes_hidden_during_analysis"]
    assert not result["estimation_prediction_replay_outcomes_hidden"]
    assert all(not hidden for _, hidden in tree.estimation_calls)


def test_estimation_accepts_zero_resolved_outcomes_without_imputing_completions(
        estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    tree.estimation_counts.update({
        "eligible_processes": 0, "completed_processes": 0, "unresolved_processes": 36,
        "resolved_configurations": 0, "unresolved_configurations": 12,
    })
    tree.release["estimation_counts"] = dict(tree.estimation_counts)
    _write_json(tree.code / "release.json", tree.release)
    _estimation_outputs(tree.code / "benchmark/estimation/results",
                        tree.estimation_counts, tree.release, tree.original)
    reproduce.main()
    result = json.loads((tree.work / "verification.json").read_text())
    assert result["estimation_counts"]["eligible_processes"] == 0
    assert result["estimation_counts"]["resolved_configurations"] == 0
    assert result["estimation_counts"]["unresolved_configurations"] == 12


def test_estimation_omits_unused_accounting_instead_of_inventing_it(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    (tree.code / "benchmark/estimation/allocation_accounting.json").unlink()
    tree.release["estimation_allocation_accounting_sha256"] = None
    _write_json(tree.code / "release.json", tree.release)
    _estimation_outputs(tree.code / "benchmark/estimation/results",
                        tree.estimation_counts, tree.release, tree.original)
    reproduce.main()
    assert not (tree.work / "artifact/benchmark/estimation/allocation_accounting.json").exists()
    assert "--allocation-accounting" not in tree.estimation_calls[1][0]


def test_estimation_wrong_frozen_seal_fails_before_any_analysis(estimation_reconstruction_tree):
    tree = estimation_reconstruction_tree
    tree.release["estimation_prediction_seal_sha256"] = "0" * 64
    _write_json(tree.code / "release.json", tree.release)
    with pytest.raises(ValueError, match="input differs from release"):
        reproduce.main()
    assert not tree.children and not tree.estimation_calls


def _host_outputs(directory, counts, release, artifact_root):
    aliases = dict(release)
    for key, value in release.items():
        if key.startswith("host_capacity_"):
            aliases[key.replace("host_capacity_", "estimation_", 1)] = value
    _estimation_outputs(directory, counts, aliases, artifact_root)
    path = directory / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance.update(
        study_freeze_sha256=release["host_capacity_study_freeze_sha256"],
        original_protocol_sha256=release["host_capacity_original_protocol_sha256"],
        original_prediction_seal_sha256=release["host_capacity_prediction_seal_sha256"])
    _write_json(path, provenance)
    _write_json(directory / "output_manifest.json", {
        p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size}
        for p in directory.iterdir() if p.is_file() and p.name != "output_manifest.json"})


@pytest.fixture
def host_reconstruction_tree(estimation_reconstruction_tree, monkeypatch):
    tree = estimation_reconstruction_tree
    tree.host_counts = dict(
        planned_processes=18, planned_pairs=9, planned_configurations=6,
        eligible_processes=9, completed_processes=0, unresolved_processes=9,
        known_launched_processes=18, unknown_launch_status_processes=0,
        resolved_configurations=3, unresolved_configurations=3)
    tree.release.update(
        expected_host_capacity_derived_files=13, host_capacity_protocol_version="host-capacity-1.0",
        host_capacity_counts=dict(tree.host_counts),
        host_capacity_original_protocol_sha256=tree.release["estimation_protocol_sha256"],
        host_capacity_prediction_seal_sha256=tree.release["estimation_prediction_seal_sha256"],
        host_capacity_execution_commits={f"host-{i}": "d" * 40 for i in range(9)})
    provenance = json.loads((tree.code / "source-provenance.json").read_text())
    for name in reproduce.HOST_CAPACITY_REQUIRED_SOURCE_FILES - set(provenance["files"]):
        path = tree.code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# authenticated synthetic host source: " + name)
        provenance["files"][name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    _write_json(tree.code / "source-provenance.json", provenance)
    frozen = tree.code / "benchmark/host_capacity"
    for name in reproduce.HOST_CAPACITY_INPUT_FILES:
        _write_json(frozen / name, {})
    _write_json(frozen / "execution_commits.json", tree.release["host_capacity_execution_commits"])
    _write_json(frozen / "allocation_accounting.json", [{"scheduler_state": "OUT_OF_MEMORY"}])
    for name, key in (
        ("protocol.json", "host_capacity_protocol_sha256"),
        ("study_freeze.json", "host_capacity_study_freeze_sha256"),
        ("execution_commits.json", "host_capacity_execution_commits_sha256"),
        ("allocation_accounting.json", "host_capacity_allocation_accounting_sha256"),
    ):
        tree.release[key] = sha256(frozen / name)
    _host_outputs(frozen / "results", tree.host_counts, tree.release, tree.original)
    _write_json(frozen / "unpermitted-outcomes.json", {"must_not_restore": True})
    _write_json(tree.code / "release.json", tree.release)
    old_extract, old_child = reproduce.extract, reproduce.run_child
    tree.host_calls, tree.host_bad_verification = [], {}
    tree.host_corrupt_output = False

    def extract_synthetic(archive, destination):
        count = old_extract(archive, destination)
        shutil.copytree(frozen, destination / "benchmark/host_capacity")
        for name in provenance["files"]:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tree.code / name, path)
        _write_json(destination / "output/estimation_host_capacity/raw.json",
                    {"raw_target_outcome": "not a fitting input"})
        return count

    def analysis_stub(arguments, work, hidden):
        artifact = work / "artifact"
        directory = artifact / "benchmark/host_capacity"
        assert not (directory / "results").exists()
        assert not (directory / "unpermitted-outcomes.json").exists()
        if arguments[0] == "-c":
            assert "memory_tuner.collect_host_capacity" in arguments[1]
            assert len(tree.estimation_calls) == 2
            tree.host_calls.append((list(arguments), list(hidden)))
            assert set(p.name for p in directory.iterdir()) == set(reproduce.HOST_CAPACITY_INPUT_FILES)
            verification = {
                "status": "passed", "prospective_outcomes_read": False, "fresh_refit_performed": False,
                **{key: tree.release["host_capacity_" + key] for key in
                   ("protocol_version", "protocol_sha256", "study_freeze_sha256",
                    "prediction_seal_sha256", "original_protocol_sha256")},
                **tree.host_bad_verification}
            _write_json(Path(arguments[-1]), verification)
        elif arguments[:2] == ["-m", "memory_tuner.collect_host_capacity"]:
            assert len(tree.host_calls) == 1
            tree.host_calls.append((list(arguments), list(hidden)))
            for option, expected in (
                ("--root", artifact), ("--source-root", artifact), ("--protocol-dir", directory),
                ("--execution-commits", directory / "execution_commits.json"),
                ("--allocation-accounting", directory / "allocation_accounting.json"),
            ):
                assert Path(arguments[arguments.index(option) + 1]) == expected
            output = Path(arguments[arguments.index("--output") + 1])
            _host_outputs(output, tree.host_counts, tree.release, artifact)
            if tree.host_corrupt_output:
                (output / "costs.json").write_text("{}")
        else:
            old_child(arguments, work, hidden)

    monkeypatch.setattr(reproduce, "extract", extract_synthetic)
    monkeypatch.setattr(reproduce, "run_child", analysis_stub)
    return tree


def test_host_is_separate_and_both_raw_cohorts_and_accounting_are_hidden_from_fitting(
        host_reconstruction_tree):
    tree = host_reconstruction_tree
    reproduce.main()
    artifact = tree.work / "artifact"
    fitting_hidden = set(tree.estimation_calls[0][1])
    for path in (
        artifact / "output/estimation_host_capacity", artifact / "output/prospective_estimation",
        artifact / "benchmark/host_capacity/allocation_accounting.json",
        artifact / "benchmark/estimation/allocation_accounting.json",
        artifact / "logs", artifact / "paper", tree.work / "reference-benchmark",
    ):
        assert path in fitting_hidden
        assert path in tree.host_calls[0][1]
    assert tree.work / "estimation-results" in tree.host_calls[0][1]
    assert artifact / "output/estimation_host_capacity" not in tree.host_calls[1][1]
    assert artifact / "benchmark/host_capacity/allocation_accounting.json" not in tree.host_calls[1][1]
    assert tree.work / "reference-benchmark" in tree.host_calls[1][1]
    result = json.loads((tree.work / "verification.json").read_text())
    assert result["host_capacity_panel_files_matched"] == 13
    assert result["host_capacity_counts"] == tree.host_counts
    assert result["estimation_counts"]["planned_processes"] == 36
    assert result["host_capacity_counts"]["planned_processes"] == 18
    assert result["host_capacity_fresh_refit_performed"] is False
    for flag in (
        "host_capacity_independent_raw_reconstruction_passed",
        "host_capacity_reference_outcomes_hidden_during_analysis",
        "host_capacity_prediction_inheritance_replay_passed",
        "host_capacity_prediction_replay_outcomes_hidden",
        "host_capacity_original_prediction_replay_passed",
        "estimation_prediction_replay_host_outcomes_hidden",
    ):
        assert result[flag] is True


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("prospective_outcomes_read", True),
    ("fresh_refit_performed", True), ("protocol_sha256", "0" * 64),
    ("study_freeze_sha256", "0" * 64), ("prediction_seal_sha256", "0" * 64),
    ("original_protocol_sha256", "0" * 64),
])
def test_host_failed_inheritance_prevents_collection_and_success(host_reconstruction_tree, field, value):
    tree = host_reconstruction_tree
    tree.host_bad_verification[field] = value
    with pytest.raises(ValueError, match="inheritance replay"):
        reproduce.main()
    assert len(tree.host_calls) == 1
    assert not (tree.work / "host-capacity-results").exists()
    assert not (tree.work / "verification.json").exists()


@pytest.mark.parametrize("mutation", ["remove_metadata", "wrong_freeze", "wrong_accounting",
                                    "wrong_count", "missing_source", "changed_source"])
def test_host_metadata_and_source_fail_before_any_analysis(host_reconstruction_tree, mutation):
    tree = host_reconstruction_tree
    if mutation == "remove_metadata":
        tree.release.pop("expected_host_capacity_derived_files")
    elif mutation == "wrong_freeze":
        tree.release["host_capacity_study_freeze_sha256"] = "0" * 64
    elif mutation == "wrong_accounting":
        tree.release["host_capacity_allocation_accounting_sha256"] = "0" * 64
    elif mutation == "wrong_count":
        tree.release["expected_host_capacity_derived_files"] = 12
    elif mutation == "missing_source":
        path = tree.code / "source-provenance.json"
        provenance = json.loads(path.read_text())
        provenance["files"].pop("memory_tuner/capture_allocation_accounting.py")
        _write_json(path, provenance)
    else:
        (tree.code / "memory_tuner/collect_host_capacity.py").write_text("# changed")
    _write_json(tree.code / "release.json", tree.release)
    with pytest.raises(ValueError):
        reproduce.main()
    assert not tree.children and not tree.host_calls
    assert not tree.estimation_calls


@pytest.mark.parametrize("mutation", ["count", "digest"])
def test_host_generated_outcomes_must_match_references(host_reconstruction_tree, mutation):
    tree = host_reconstruction_tree
    if mutation == "count":
        tree.host_counts["eligible_processes"] = 18
    else:
        tree.host_corrupt_output = True
    with pytest.raises(ValueError):
        reproduce.main()
    assert not (tree.work / "verification.json").exists()


def test_host_nonisolated_replay_does_not_claim_hidden_evidence(host_reconstruction_tree):
    tree = host_reconstruction_tree
    tree.argv.remove("--isolate-analysis")
    reproduce.main()
    result = json.loads((tree.work / "verification.json").read_text())
    assert result["host_capacity_independent_raw_reconstruction_passed"]
    assert not result["host_capacity_reference_outcomes_hidden_during_analysis"]
    assert not result["host_capacity_prediction_replay_outcomes_hidden"]
    assert not result["estimation_prediction_replay_host_outcomes_hidden"]


def test_real_namespace_hides_accounting_file_but_preserves_input_and_writable_tmp(tmp_path, monkeypatch):
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap unavailable")
    probe = subprocess.run([bwrap, "--ro-bind", "/", "/", "--", "/usr/bin/true"],
                           capture_output=True, timeout=10)
    if probe.returncode:
        pytest.skip("bubblewrap namespace unavailable")
    work, code = tmp_path / "work", tmp_path / "code"
    work.mkdir()
    code.mkdir()
    accounting = work / "allocation_accounting.json"
    accounting.write_text('{"scheduler_state":"OUT_OF_MEMORY"}')
    before = accounting.read_bytes()
    monkeypatch.setattr(reproduce, "ROOT", code)
    reproduce.run_child(["-c",
        "import tempfile; from pathlib import Path; "
        f"assert Path({str(accounting)!r}).read_bytes()==b''; "
        "f=tempfile.TemporaryFile(); f.write(b'works'); f.close()"], work, [accounting])
    assert accounting.read_bytes() == before
