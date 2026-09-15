#!/usr/bin/env python3
"""Rebuild frozen RLVRAMBench results from an integrity-checked public archive."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESTIMATION_INPUT_FILES = (
    "protocol.json", "amendment-01.json", "matrix.csv", "model_metadata.json",
    "targets.csv", "fitted_model.json", "predictions.json", "retrospective.json",
    "prediction_freeze.json", "execution_commits.json",
)
ESTIMATION_COUNT_FIELDS = (
    "planned_processes", "planned_pairs", "planned_configurations",
    "eligible_processes", "completed_processes", "unresolved_processes",
    "known_launched_processes", "unknown_launch_status_processes",
    "resolved_configurations",
)
ESTIMATION_REQUIRED_SOURCE_FILES = {
    "benchmark.py", "memory_tuner/estimation_baselines.py",
    "memory_tuner/verify_estimation.py", "memory_tuner/collect_estimation_study.py",
    "memory_tuner/model_memory_metadata.py", "memory_tuner/plan_estimation_study.py",
    "memory_tuner/run_matched_gpu.py", "memory_tuner/device_contract.py",
    "memory_tuner/vllm_uuid_compat.py", "memory_tuner/benchmark_v2_env.py",
    "memory_tuner/resolve_hf_snapshot.py", "memory_tuner/grpo_raw_evidence.py",
    "memory_tuner/scientific_revision_analysis.py",
    "instrumented_python_packages/sitecustomize.py",
    "instrumented_python_packages/verl/__init__.py",
}
HOST_CAPACITY_INPUT_FILES = (
    "protocol.json", "study_freeze.json", "matrix.csv", "targets.csv", "predictions.json",
    "execution_commits.json", "allocation_accounting.json",
)
HOST_CAPACITY_REQUIRED_SOURCE_FILES = ESTIMATION_REQUIRED_SOURCE_FILES | {
    "memory_tuner/capture_allocation_accounting.py",
    "memory_tuner/host_capacity_protocol.py", "memory_tuner/test_host_capacity_protocol.py",
    "memory_tuner/collect_host_capacity.py", "memory_tuner/test_collect_host_capacity.py",
    "memory_tuner/run_host_capacity.py", "memory_tuner/test_run_host_capacity.py",
    "memory_tuner/test_capture_allocation_accounting.py", "run_host_capacity.slurm",
}
HOST_CAPACITY_OUTPUT_FILES = {
    f"{name}.json" for name in
    ("processes", "configurations", "configurations_flat", "pairs", "evaluation",
     "summary", "costs", "provenance", "output_manifest")
} | {f"{name}.csv" for name in ("processes", "configurations", "configurations_flat", "pairs")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_member(member: tarfile.TarInfo, destination: Path) -> None:
    relative = Path(member.name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {member.name}")
    path = destination / relative
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Archive member escapes destination: {member.name}")
    if member.issym() or member.islnk():
        link = Path(member.linkname)
        if link.is_absolute():
            raise ValueError(f"Absolute archive link: {member.name}")
        target = (path.parent if member.issym() else destination) / link
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"Archive link escapes destination: {member.name}")
    elif not (member.isfile() or member.isdir()):
        raise ValueError(f"Unsupported archive member type: {member.name}")


def extract(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    decoder = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as contents:
            for member in contents:
                check_member(member, destination)
                contents.extract(member, destination, set_attrs=False)
                count += 1
        decoder.stdout.close()
        if decoder.wait() != 0:
            raise RuntimeError("zstd decompression failed")
    finally:
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait()
    return count


def normalize_paths(text: str, roots: list[Path]) -> str:
    for root in sorted(roots, key=lambda value: len(str(value)), reverse=True):
        text = text.replace(str(root), "<ARTIFACT_ROOT>")
    return text


def run_child(arguments, work: Path, hidden: list[Path]) -> None:
    work = work.resolve()
    temporary = work / "tmp"
    if temporary.is_symlink():
        raise ValueError("Child temporary directory must not be a symlink")
    temporary.mkdir(mode=0o700, exist_ok=True)
    if temporary.resolve() != temporary:
        raise ValueError("Child temporary directory escapes the work-local path")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(work / "cache/matplotlib"),
        "XDG_CACHE_HOME": str(work / "cache"),
        "TMPDIR": str(temporary),
    })
    command = [sys.executable, *arguments]
    if hidden:
        if shutil.which("bwrap") is None:
            raise RuntimeError("--isolate-analysis requires bubblewrap (bwrap)")
        wrapper = [
            "bwrap", "--die-with-parent", "--ro-bind", "/", "/",
            "--bind", str(work), str(work),
        ]
        # Keep a project-local environment usable even when its project is hidden.
        # This read-only mount exposes only the interpreter and dependencies.
        isolated_python = "/run/rlvrambench-python"
        wrapper += ["--tmpfs", "/run", "--ro-bind", sys.prefix, isolated_python]
        for path in hidden:
            if path.exists():
                if path.is_dir():
                    wrapper += ["--tmpfs", str(path)]
                elif path.is_file():
                    # A /dev/null bind can be unreadable on nodev filesystems.
                    # Use a validated empty regular file in the safe local tmp.
                    mask = temporary / "hidden-file.empty"
                    if mask.is_symlink():
                        raise ValueError("Hidden-file mask must not be a symlink")
                    if mask.exists():
                        if not mask.is_file() or mask.stat().st_size != 0:
                            raise ValueError("Hidden-file mask is not an empty regular file")
                    else:
                        with mask.open("xb"):
                            pass
                    wrapper += ["--ro-bind", str(mask), str(path)]
                else:
                    raise ValueError(f"Unsupported hidden evidence path: {path}")
        command[0] = isolated_python + "/bin/python"
        # Probe the actual child namespace, not merely the requested option.
        probe = (
            "import json,sys; from pathlib import Path; "
            "paths=[Path(p) for p in json.loads(sys.argv[1])]; "
            "assert all(not p.exists() or "
            "(not any(p.iterdir()) if p.is_dir() else p.read_bytes() == b'') for p in paths), "
            "'Reference evidence remains visible in the child namespace'"
        )
        subprocess.run(
            wrapper + ["--", command[0], "-c", probe,
                       json.dumps([str(p) for p in hidden])],
            cwd=ROOT, env=env, check=True,
        )
        command = wrapper + ["--", *command]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def verify_estimation_source(artifact: Path, release: dict, *, required=None) -> int:
    """The executed standalone modules must match their archived source copies."""
    provenance = json.loads((ROOT / "source-provenance.json").read_text())
    entries = provenance["files"]
    required = set(ESTIMATION_REQUIRED_SOURCE_FILES if required is None else required)
    if (provenance["source_commit"] != release["source_commit"]
            or not required <= set(entries)):
        raise ValueError("Estimation source provenance lacks the committed analysis closure")
    # Earlier estimator releases did not import this helper. Keep those valid,
    # but never execute the new dependency without archived source hashes.
    module = ast.parse((ROOT / "memory_tuner/collect_estimation_study.py").read_text())
    if any(isinstance(node, ast.ImportFrom) and node.module == "memory_tuner"
           and any(alias.name == "capture_allocation_accounting" for alias in node.names)
           for node in ast.walk(module)):
        if "memory_tuner/capture_allocation_accounting.py" not in entries:
            raise ValueError("Estimation source provenance lacks the accounting analysis closure")
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source-provenance path: {name}")
        for directory in (ROOT, artifact):
            path = directory / relative
            if (not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
                    or sha256(path) != expected["sha256"]
                    or path.stat().st_size != expected["size_bytes"]):
                raise ValueError(f"Estimation analysis source differs: {directory / relative}")
    return len(entries)


def restore_estimation_inputs(reference: Path, artifact: Path, release: dict) -> Path:
    source = reference / "estimation"
    destination = artifact / "benchmark/estimation"
    destination.mkdir(parents=True, exist_ok=False)
    names = list(ESTIMATION_INPUT_FILES)
    if release.get("estimation_allocation_accounting_sha256") is not None:
        names.append("allocation_accounting.json")
    for name in names:
        path = source / name
        if not path.is_file() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Missing or external frozen estimation input: {name}")
        shutil.copy2(path, destination / name)
    for name, key in (
        ("protocol.json", "estimation_protocol_sha256"),
        ("prediction_freeze.json", "estimation_prediction_seal_sha256"),
        ("execution_commits.json", "estimation_execution_commits_sha256"),
    ):
        if sha256(destination / name) != release[key]:
            raise ValueError(f"Estimation input differs from release: {name}")
    if ("allocation_accounting.json" in names
            and sha256(destination / "allocation_accounting.json")
            != release["estimation_allocation_accounting_sha256"]):
        raise ValueError("Estimation allocation accounting differs from release")
    if (json.loads((destination / "execution_commits.json").read_text())
            != release["estimation_execution_commits"]):
        raise ValueError("Estimation execution-commit map differs from release")
    return destination


def restore_host_capacity_inputs(reference: Path, artifact: Path, release: dict) -> Path:
    source, destination = reference / "host_capacity", artifact / "benchmark/host_capacity"
    destination.mkdir(parents=True, exist_ok=False)
    for name in HOST_CAPACITY_INPUT_FILES:
        path = source / name
        if not path.is_file() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Missing or external frozen host-capacity input: {name}")
        shutil.copy2(path, destination / name)
    for name, key in (
        ("protocol.json", "host_capacity_protocol_sha256"),
        ("study_freeze.json", "host_capacity_study_freeze_sha256"),
        ("execution_commits.json", "host_capacity_execution_commits_sha256"),
        ("allocation_accounting.json", "host_capacity_allocation_accounting_sha256"),
    ):
        if sha256(destination / name) != release[key]:
            raise ValueError(f"Host-capacity input differs from release: {name}")
    if (json.loads((destination / "execution_commits.json").read_text())
            != release["host_capacity_execution_commits"]
            or release["host_capacity_prediction_seal_sha256"]
            != release["estimation_prediction_seal_sha256"]
            or release["host_capacity_original_protocol_sha256"] != release["estimation_protocol_sha256"]):
        raise ValueError("Host-capacity maps or original predictor anchors differ from release")
    return destination


def prediction_replay_hidden_paths(artifact: Path, work: Path, hidden: list[Path]) -> list[Path]:
    """Accounting contains post-execution states, so hide files as well as raw trees."""
    return list(dict.fromkeys([
        *hidden, artifact / "output/prospective_estimation",
        artifact / "output/estimation_host_capacity",
        artifact / "output/estimation-device-preflight", artifact / "logs", artifact / "paper",
        artifact / "benchmark/estimation/allocation_accounting.json",
        artifact / "benchmark/host_capacity/allocation_accounting.json",
        work / "estimation-results",
    ])) if hidden else []


def estimation_output_inventory(directory: Path) -> dict:
    manifest = json.loads((directory / "output_manifest.json").read_text())
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Missing estimation collector output inventory")
    for name, expected in manifest.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or name == "output_manifest.json" or relative.suffix not in (".json", ".csv")):
            raise ValueError(f"Unsafe estimation output: {name}")
        path = directory / relative
        if (not path.is_file() or not path.resolve().is_relative_to(directory.resolve())
                or sha256(path) != expected["sha256"]
                or path.stat().st_size != expected["size_bytes"]):
            raise ValueError(f"Estimation output manifest mismatch: {name}")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest) | {"output_manifest.json"}:
        raise ValueError("Estimation output inventory is incomplete")
    return manifest


def reconstruct_estimation(artifact: Path, reference: Path, curated: Path,
                           work: Path, hidden: list[Path], release: dict,
                           original_root: Path) -> dict:
    count = release["expected_estimation_derived_files"]
    source_count = verify_estimation_source(artifact, release)
    # Only independently regenerated historical tables become fitting inputs.
    for path in curated.iterdir():
        if path.is_file():
            shutil.copy2(path, artifact / "benchmark" / path.name)
    directory = artifact / "benchmark/estimation"
    if (directory / "results").exists():
        raise ValueError("Published estimation outcomes leaked into the analysis tree")
    verification_path = work / "estimation-verification.json"
    # Replay/fitting must not see even the prospective raw outcomes or scheduler
    # logs. The collector later needs those raw trees, but never their references.
    replay_hidden = prediction_replay_hidden_paths(artifact, work, hidden)
    run_child([
        "-m", "memory_tuner.verify_estimation", "--root", str(artifact),
        "--refit", "--output", str(verification_path),
    ], work, replay_hidden)
    verification = json.loads(verification_path.read_text())
    if (verification["status"] != "passed"
            or verification["prediction_seal_sha256"] != release["estimation_prediction_seal_sha256"]
            or verification["protocol_sha256"] != release["estimation_protocol_sha256"]
            or verification["prospective_outcomes_read"] is not False
            or verification["fresh_refit_compared"] is not True):
        raise ValueError("Independent estimation prediction replay/refit has not passed")
    output = work / "estimation-results"
    arguments = [
        "-m", "memory_tuner.collect_estimation_study",
        "--root", str(artifact), "--source-root", str(artifact),
        "--protocol-dir", str(directory),
        "--execution-commits", str(directory / "execution_commits.json"),
        "--output", str(output),
        "--protocol-sha256", release["estimation_protocol_sha256"],
        "--prediction-seal-sha256", release["estimation_prediction_seal_sha256"],
    ]
    if release.get("estimation_allocation_accounting_sha256") is not None:
        arguments += ["--allocation-accounting", str(directory / "allocation_accounting.json")]
    run_child(arguments, work, hidden)
    expected = estimation_output_inventory(reference / "estimation/results")
    actual = estimation_output_inventory(output)
    if set(actual) != set(expected) or len(actual) + 1 != count:
        raise ValueError("Estimation reconstructed file set differs from release")
    for name in expected:
        old = normalize_paths((reference / "estimation/results" / name).read_text(),
                              [original_root, artifact])
        new = normalize_paths((output / name).read_text(), [original_root, artifact])
        if old != new:
            raise ValueError(f"Estimation reconstruction differs: {name}")
    # Each output manifest was checked against its own bytes above. Rebased
    # path strings can change those byte hashes without changing the evidence.
    summary = json.loads((output / "summary.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    counts = {name: summary[name] for name in ESTIMATION_COUNT_FIELDS}
    counts["unresolved_configurations"] = evaluation["unresolved_configurations"]
    if (any(type(value) is not int or value < 0 for value in counts.values())
            or json.dumps(counts, sort_keys=True) != json.dumps(release["estimation_counts"], sort_keys=True)
            or provenance["execution_commits_expected"] != release["estimation_execution_commits"]
            or provenance["prediction_seal_sha256"] != release["estimation_prediction_seal_sha256"]
            or provenance["protocol_sha256"] != release["estimation_protocol_sha256"]
            or provenance["execution_commits_file_sha256"]
            != release["estimation_execution_commits_sha256"]
            or provenance.get("allocation_accounting_sha256")
            != release.get("estimation_allocation_accounting_sha256")):
        raise ValueError("Estimation reconstructed counts or provenance differ from release")
    return {
        "estimation_panel_files_matched": len(actual) + 1,
        "estimation_counts": counts,
        "estimation_prediction_seal_sha256": verification["prediction_seal_sha256"],
        "estimation_execution_commits": provenance["execution_commits_expected"],
        "estimation_independent_raw_reconstruction_passed": True,
        "estimation_reference_outcomes_hidden_during_analysis": bool(hidden),
        "estimation_prediction_replay_passed": True,
        "estimation_prediction_replay_outcomes_hidden": bool(replay_hidden),
        "estimation_prediction_replay_host_outcomes_hidden": bool(replay_hidden),
        "estimation_fresh_refit_compared": verification["fresh_refit_compared"],
        "estimation_source_files_verified": source_count,
        "estimation_output_comparison": (
            "Exact file sets and contents after artifact-root path normalization; "
            "each output manifest independently verified against its own bytes."),
    }


def reconstruct_host_capacity(artifact: Path, reference: Path, work: Path,
                              hidden: list[Path], release: dict, original_root: Path,
                              estimation_result: dict) -> dict:
    """Validate exact frozen inheritance, then collect the separate host raw cohort."""
    count = release["expected_host_capacity_derived_files"]
    if (type(count) is not int or count != len(HOST_CAPACITY_OUTPUT_FILES)
            or estimation_result.get("estimation_prediction_replay_passed") is not True
            or estimation_result.get("estimation_prediction_seal_sha256")
            != release["host_capacity_prediction_seal_sha256"]):
        raise ValueError("Host-capacity needs its complete inventory and prior original prediction replay")
    source_count = verify_estimation_source(
        artifact, release, required=HOST_CAPACITY_REQUIRED_SOURCE_FILES)
    directory = artifact / "benchmark/host_capacity"
    if (directory / "results").exists():
        raise ValueError("Published host-capacity outcomes leaked into the analysis tree")
    verification_path = work / "host-capacity-inheritance-verification.json"
    replay_hidden = prediction_replay_hidden_paths(artifact, work, hidden)
    # This archived loader has no fitting call. It checks original sealed
    # inputs, exact inheritance, active/source hashes and the new chronology.
    script = (
        "import json,sys; from pathlib import Path; "
        "from memory_tuner.collect_estimation_study import Evidence; "
        "from memory_tuner.collect_host_capacity import load_frozen; "
        "f=load_frozen(Evidence(Path(sys.argv[1])), sys.argv[2], sys.argv[3]); "
        "p=f['study_provenance']; "
        "assert p['study_freeze_sha256']==sys.argv[4], 'Host study freeze differs'; "
        "r=dict(status='passed', protocol_version=f['protocol']['protocol_version'], "
        "protocol_sha256=f['protocol_sha256'], study_freeze_sha256=p['study_freeze_sha256'], "
        "prediction_seal_sha256=f['prediction_seal_sha256'], "
        "original_protocol_sha256=p['original_protocol_sha256'], "
        "prospective_outcomes_read=False, fresh_refit_performed=False); "
        "h=Path(sys.argv[5]).open('x'); json.dump(r,h,sort_keys=True); h.close()"
    )
    run_child([
        "-c", script, str(artifact), release["host_capacity_protocol_sha256"],
        release["host_capacity_prediction_seal_sha256"],
        release["host_capacity_study_freeze_sha256"], str(verification_path),
    ], work, replay_hidden)
    verification = json.loads(verification_path.read_text())
    if (verification.get("status") != "passed"
            or verification.get("prospective_outcomes_read") is not False
            or verification.get("fresh_refit_performed") is not False
            or any(verification.get(key) != release["host_capacity_" + key]
                   for key in ("protocol_version", "protocol_sha256", "study_freeze_sha256",
                               "prediction_seal_sha256", "original_protocol_sha256"))):
        raise ValueError("Independent host-capacity frozen inheritance replay has not passed")
    output = work / "host-capacity-results"
    collector_hidden = list(dict.fromkeys([
        *hidden, work / "estimation-results", artifact / "output/prospective_estimation",
    ])) if hidden else []
    run_child([
        "-m", "memory_tuner.collect_host_capacity", "--root", str(artifact),
        "--source-root", str(artifact), "--protocol-dir", str(directory),
        "--execution-commits", str(directory / "execution_commits.json"),
        "--allocation-accounting", str(directory / "allocation_accounting.json"),
        "--protocol-sha256", release["host_capacity_protocol_sha256"],
        "--prediction-seal-sha256", release["host_capacity_prediction_seal_sha256"],
        "--output", str(output),
    ], work, collector_hidden)
    expected = estimation_output_inventory(reference / "host_capacity/results")
    actual = estimation_output_inventory(output)
    if (set(actual) != set(expected) or set(actual) | {"output_manifest.json"}
            != HOST_CAPACITY_OUTPUT_FILES or len(actual) + 1 != count):
        raise ValueError("Host-capacity reconstructed file set differs from release")
    for name in expected:
        old = normalize_paths((reference / "host_capacity/results" / name).read_text(),
                              [original_root, artifact])
        new = normalize_paths((output / name).read_text(), [original_root, artifact])
        if old != new:
            raise ValueError(f"Host-capacity reconstruction differs: {name}")
    summary = json.loads((output / "summary.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    counts = {name: summary[name] for name in ESTIMATION_COUNT_FIELDS}
    counts["unresolved_configurations"] = evaluation["unresolved_configurations"]
    if (any(type(value) is not int or value < 0 for value in counts.values())
            or json.dumps(counts, sort_keys=True) != json.dumps(release["host_capacity_counts"], sort_keys=True)
            or any(provenance[key] != release["host_capacity_" + key] for key in
                   ("protocol_sha256", "study_freeze_sha256", "prediction_seal_sha256",
                    "original_protocol_sha256", "allocation_accounting_sha256"))
            or provenance["original_prediction_seal_sha256"] != release["host_capacity_prediction_seal_sha256"]
            or provenance["execution_commits_expected"] != release["host_capacity_execution_commits"]
            or provenance["execution_commits_file_sha256"] != release["host_capacity_execution_commits_sha256"]):
        raise ValueError("Host-capacity reconstructed counts or provenance differ from release")
    return {
        "host_capacity_panel_files_matched": len(actual) + 1, "host_capacity_counts": counts,
        **{"host_capacity_" + key: verification[key] for key in
           ("protocol_version", "protocol_sha256", "study_freeze_sha256",
            "prediction_seal_sha256", "original_protocol_sha256")},
        **{key: release[key] for key in (
            "host_capacity_execution_commits", "host_capacity_execution_commits_sha256",
            "host_capacity_allocation_accounting_sha256")},
        "host_capacity_independent_raw_reconstruction_passed": True,
        "host_capacity_reference_outcomes_hidden_during_analysis": bool(collector_hidden),
        "host_capacity_prediction_inheritance_replay_passed": True,
        "host_capacity_prediction_replay_outcomes_hidden": bool(replay_hidden),
        "host_capacity_original_prediction_replay_passed": True,
        "host_capacity_fresh_refit_performed": False, "host_capacity_source_files_verified": source_count,
        "host_capacity_output_comparison": (
            "Separate exact file sets and contents after artifact-root path normalization; "
            "each output manifest independently checked against its own bytes."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="New, nonexistent output directory; existing data are never overwritten")
    parser.add_argument("--isolate-analysis", action="store_true",
                        help="Hide reference tables and the original project during analysis")
    parser.add_argument("--hide-path", type=Path, action="append", default=[],
                        help="Additional known evidence/implementation directory to hide; repeatable")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use CPython 3.11 with publication-requirements.txt")
    release = json.loads((ROOT / "release.json").read_text())
    from memory_tuner.artifact_paths import ORIGINAL_ROOT
    if sha256(args.archive) != release["archive"]["sha256"]:
        raise SystemExit("Evidence archive SHA-256 does not match release.json")
    work = args.work_dir.resolve()
    extra_hidden = [p.resolve() for p in args.hide_path]
    if extra_hidden and not args.isolate_analysis:
        raise ValueError("--hide-path requires --isolate-analysis")
    for path in ([ORIGINAL_ROOT, ROOT / "benchmark", *extra_hidden]
                 if args.isolate_analysis else []):
        if not path.is_dir():
            if path in extra_hidden:
                raise ValueError(f"Additional hidden path is not a directory: {path}")
            continue
        if work.is_relative_to(path) or ROOT.is_relative_to(path):
            raise ValueError("Reconstruction work/code directory must be outside hidden paths")
    work.mkdir(parents=True, exist_ok=False)
    artifact = work / "artifact"
    members = extract(args.archive.resolve(), artifact)

    from memory_tuner.artifact_manifest import read_csv, verify_manifest
    manifest = artifact / "profiles/benchmark/publication-artifact-manifest.csv"
    errors = verify_manifest(artifact, manifest)
    if errors:
        raise SystemExit("Extracted evidence verification failed:\n" + "\n".join(errors))
    manifest_entries = len(read_csv(manifest))
    manifest_digest = sha256(manifest)
    print(f"Verified {manifest_entries} evidence files.", flush=True)
    estimation_count = release.get("expected_estimation_derived_files", 0)
    has_estimation_reference = (artifact / "benchmark/estimation/results/summary.json").is_file()
    if estimation_count or has_estimation_reference:
        if (type(estimation_count) is not int or estimation_count <= 0
                or not has_estimation_reference or not release.get("benchmark_protocol_version")):
            raise ValueError("Estimation release metadata and collector references disagree")
        verify_estimation_source(artifact, release)
    host_capacity_count = release.get("expected_host_capacity_derived_files", 0)
    has_host_capacity_reference = (artifact / "benchmark/host_capacity/results/summary.json").is_file()
    if host_capacity_count or has_host_capacity_reference:
        if (type(host_capacity_count) is not int or host_capacity_count != len(HOST_CAPACITY_OUTPUT_FILES)
                or not has_host_capacity_reference or not estimation_count):
            raise ValueError("Host-capacity release metadata and collector references disagree")
        verify_estimation_source(artifact, release, required=HOST_CAPACITY_REQUIRED_SOURCE_FILES)

    # Keep the authenticated reference tables for comparison, never as analysis inputs.
    reference = work / "reference-profiles"
    (artifact / "profiles").rename(reference)
    curated_reference = work / "reference-benchmark"
    if release.get("benchmark_protocol_version"):
        (artifact / "benchmark").rename(curated_reference)
    decision_count = int(release.get("expected_decision_derived_files", 0))
    if decision_count:
        # Frozen specifications are inputs; never expose the published target
        # outcomes or replay outputs to their own reconstruction.
        decision_inputs = artifact / "benchmark/decision"
        decision_inputs.mkdir(parents=True)
        for name in ("protocol.json", "matrix.csv", "amendment-01.json",
                     "submission.json", "repairs.json"):
            source = curated_reference / "decision" / name
            if source.is_file():
                shutil.copy2(source, decision_inputs / name)
    if estimation_count:
        restore_estimation_inputs(curated_reference, artifact, release)
    if host_capacity_count:
        restore_host_capacity_inputs(curated_reference, artifact, release)
    shutil.copytree(
        reference / "strengthening/same_node",
        artifact / "profiles/strengthening/same_node",
    )
    hidden = list(dict.fromkeys([
        reference, curated_reference, ORIGINAL_ROOT, ROOT / "benchmark", *extra_hidden,
    ])) if args.isolate_analysis else []
    tables = work / "results"
    report = work / "reports/evidence.md"
    run_child([
        "-m", "memory_tuner.standard_grpo_paper_analysis",
        "--root", str(artifact), "--output-dir", str(tables), "--report", str(report),
    ], work, hidden)

    originals = sorted(
        path for path in (reference / "standard_grpo").iterdir()
        if path.suffix in {".csv", ".json"}
    )
    if len(originals) != release["expected_derived_files"]:
        raise ValueError("Unexpected frozen reference table count")
    generated = {path.name for path in tables.iterdir() if path.suffix in {".csv", ".json"}}
    if generated != {path.name for path in originals}:
        raise ValueError("Regenerated table file set differs from the frozen release")
    for expected in originals:
        old = normalize_paths(expected.read_text(), [ORIGINAL_ROOT, artifact])
        new = normalize_paths((tables / expected.name).read_text(), [ORIGINAL_ROOT, artifact])
        if old != new:
            raise ValueError(f"Regenerated results differ: {expected.name}")
    summary = json.loads((tables / "summary.json").read_text())
    if summary["scope"]["principal_fresh_processes"] != 588:
        raise ValueError("The complete 588-process corpus was not reconstructed")
    if summary["scope"]["supporting_historical_source_processes"] != 16:
        raise ValueError("Historical source-screen count differs")
    review_count = int(release.get("expected_review_derived_files", 0))
    if review_count:
        # These inputs are our regenerated tables, not the hidden references.
        shutil.copytree(tables, artifact / "profiles/standard_grpo")
        review_tables = work / "review-results"
        for module in (
            "review_evidence_analysis", "review_attempt_workload_audit",
            "review_control_analysis",
        ):
            run_child([
                "-m", f"memory_tuner.{module}", "--root", str(artifact),
                "--output", str(review_tables),
            ], work, hidden)
        expected_review = sorted((reference / "review_revision").glob("*"))
        expected_review = [p for p in expected_review if p.suffix in (".csv", ".json")]
        if len(expected_review) != review_count:
            raise ValueError("Unexpected revised reference-table count")
        if {p.name for p in review_tables.iterdir()} != {p.name for p in expected_review}:
            raise ValueError("Revised result file set differs from its frozen release")
        for expected in expected_review:
            old = normalize_paths(expected.read_text(), [ORIGINAL_ROOT, artifact])
            new = normalize_paths((review_tables / expected.name).read_text(),
                                  [ORIGINAL_ROOT, artifact])
            if old != new:
                raise ValueError(f"Revised reconstruction differs: {expected.name}")
        control = json.loads((review_tables / "control-summary.json").read_text())
        if (control["completed"], control["attempts"]) != (
                release["review_control_completed_processes"],
                release["review_control_attempted_processes"]):
            raise ValueError("Revised control attempt/completion accounting differs")
        run_child([
            "-m", "memory_tuner.make_review_figures", "--root", str(artifact),
            "--review-results", str(review_tables), "--output", str(work / "figures"),
        ], work, hidden)
        figure_names = (
            "standard_grpo_boundary", "standard_grpo_mechanisms",
            "standard_grpo_temporal", "review_instrumentation_batch",
        )
    else:
        run_child([
            "-m", "memory_tuner.make_standard_grpo_figures",
            "--data", str(tables), "--output", str(work / "figures"),
        ], work, hidden)
        figure_names = (
            "standard_grpo_boundary", "standard_grpo_mechanisms",
            "standard_grpo_temporal", "standard_grpo_phase_effects",
        )
    for name in figure_names:
        for extension in (".pdf", ".png"):
            if (work / "figures" / (name + extension)).stat().st_size == 0:
                raise ValueError(f"Empty generated figure: {name}{extension}")
    curated_count = 0
    estimation_result = {}
    host_capacity_result = {}
    if release.get("benchmark_protocol_version"):
        shutil.copytree(review_tables, artifact / "profiles/review_revision")
        curated = work / "benchmark"
        run_child([
            str(ROOT / "export_benchmark.py"), "--root", str(artifact),
            "--output", str(curated),
        ], work, hidden)
        expected_names = {p.name for p in curated_reference.iterdir() if p.is_file()}
        actual_names = {p.name for p in curated.iterdir() if p.is_file()}
        if actual_names != expected_names:
            raise ValueError("Curated benchmark file set differs")
        for name in sorted(expected_names):
            if (curated / name).read_bytes() != (curated_reference / name).read_bytes():
                raise ValueError(f"Reconstructed curated benchmark differs: {name}")
        curated_count = len(expected_names)
    if estimation_count:
        estimation_result = reconstruct_estimation(
            artifact, curated_reference, curated, work, hidden, release, ORIGINAL_ROOT)
    if host_capacity_count:
        host_capacity_result = reconstruct_host_capacity(
            artifact, curated_reference, work, hidden, release, ORIGINAL_ROOT, estimation_result)
    if decision_count:
        decision_output = work / "decision-results"
        run_child([
            "-m", "memory_tuner.collect_decision_benchmark",
            "--root", str(artifact),
            "--protocol-dir", str(artifact / "benchmark/decision"),
            "--output", str(decision_output),
        ], work, hidden)
        decision_reference = curated_reference / "decision/results"
        expected_names = {p.name for p in decision_reference.iterdir() if p.is_file()}
        actual_names = {p.name for p in decision_output.iterdir() if p.is_file()}
        if len(expected_names) != decision_count or actual_names != expected_names:
            raise ValueError("Decision-panel output file set differs")
        for name in sorted(expected_names):
            if (decision_output / name).read_bytes() != (decision_reference / name).read_bytes():
                raise ValueError(f"Reconstructed decision-panel output differs: {name}")
        decision_summary = json.loads(
            (decision_output / "collection_summary.json").read_text())
        if (decision_summary["planned_slots"] != release["decision_planned_slots"]
                or decision_summary["recorded_attempts"] != release["decision_recorded_attempts"]
                or decision_summary["validated_memory_outcomes"] !=
                release["decision_validated_memory_outcomes"]):
            raise ValueError("Decision-panel slot/attempt accounting differs")
    result = {
        "status": "passed", "archive_sha256": sha256(args.archive),
        "archive_members": members, "manifest_entries_verified": manifest_entries,
        "evidence_manifest_sha256": manifest_digest,
        "principal_processes": 588, "supporting_source_processes": 16,
        "derived_csv_json_files_matched": len(originals), "figures_regenerated": 4,
        "revised_csv_json_files_matched": review_count,
        "curated_benchmark_files_matched": curated_count,
        "decision_panel_files_matched": decision_count,
        "decision_validated_memory_outcomes": (
            decision_summary["validated_memory_outcomes"] if decision_count else 0),
        "decision_reference_outcomes_hidden_during_analysis": bool(
            decision_count and args.isolate_analysis),
        "curated_references_hidden_during_analysis": bool(curated_count and args.isolate_analysis),
        "review_control_completed_processes": release.get("review_control_completed_processes", 0),
        "review_control_attempted_processes": release.get("review_control_attempted_processes", 0),
        "reference_profiles_removed_from_analysis_root": True,
        "original_project_and_reference_tables_hidden": args.isolate_analysis,
        "hidden_paths": [str(p) for p in hidden],
        "child_namespace_reference_inaccessibility_checked": bool(hidden),
        "network_namespace_isolation_claimed": False,
        **estimation_result,
        **host_capacity_result,
    }
    (work / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
