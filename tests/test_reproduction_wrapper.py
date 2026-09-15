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


def test_run_child_hides_references_in_real_bubblewrap_namespace(tmp_path, monkeypatch):
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
    script = (
        "from pathlib import Path; "
        f"paths={json.dumps([str(p) for p in hidden])}; "
        "assert all(not any(Path(p).iterdir()) for p in paths); "
        f"Path({str(marker)!r}).write_text('passed')"
    )
    monkeypatch.setattr(reproduce, "ROOT", code)
    monkeypatch.setattr(reproduce.os, "environ", {"PATH": os.defpath})
    reproduce.run_child(["-c", script], work, hidden)
    assert marker.read_text() == "passed"
    assert all((path / "outcomes.csv").is_file() for path in hidden)
