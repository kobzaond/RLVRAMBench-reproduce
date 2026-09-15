#!/usr/bin/env python3
"""Rebuild frozen RLVRAMBench results from an integrity-checked public archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(work / "cache/matplotlib"),
        "XDG_CACHE_HOME": str(work / "cache"),
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
                wrapper += ["--tmpfs", str(path)]
        command[0] = isolated_python + "/bin/python"
        # Probe the actual child namespace, not merely the requested option.
        probe = (
            "import json,sys; from pathlib import Path; "
            "paths=[Path(p) for p in json.loads(sys.argv[1])]; "
            "assert all(not p.exists() or not any(p.iterdir()) for p in paths), "
            "'Reference evidence remains visible in the child namespace'"
        )
        subprocess.run(
            wrapper + ["--", command[0], "-c", probe,
                       json.dumps([str(p) for p in hidden])],
            cwd=ROOT, env=env, check=True,
        )
        command = wrapper + ["--", *command]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


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
    }
    (work / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
