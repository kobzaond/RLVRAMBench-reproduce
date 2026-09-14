#!/usr/bin/env python3
"""Rebuild the frozen RLVRAMBench results from an authenticated evidence archive."""
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
        for path in hidden:
            if path.exists():
                wrapper += ["--tmpfs", str(path)]
        command = wrapper + ["--", *command]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="New, nonexistent output directory; existing data are never overwritten")
    parser.add_argument("--isolate-analysis", action="store_true",
                        help="Hide reference tables and the original project during analysis")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use CPython 3.11 with the pinned requirements.txt")
    release = json.loads((ROOT / "release.json").read_text())
    if sha256(args.archive) != release["archive"]["sha256"]:
        raise SystemExit("Evidence archive SHA-256 does not match release.json")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    artifact = work / "artifact"
    members = extract(args.archive.resolve(), artifact)

    from memory_tuner.artifact_manifest import read_csv, verify_manifest
    from memory_tuner.artifact_paths import ORIGINAL_ROOT

    manifest = artifact / "profiles/benchmark/publication-artifact-manifest.csv"
    errors = verify_manifest(artifact, manifest)
    if errors:
        raise SystemExit("Extracted evidence verification failed:\n" + "\n".join(errors))
    manifest_entries = len(read_csv(manifest))
    print(f"Verified {manifest_entries} evidence files.", flush=True)

    # Keep the authenticated reference tables for comparison, never as analysis inputs.
    reference = work / "reference-profiles"
    (artifact / "profiles").rename(reference)
    shutil.copytree(
        reference / "strengthening/same_node",
        artifact / "profiles/strengthening/same_node",
    )
    hidden = [reference, ORIGINAL_ROOT] if args.isolate_analysis else []
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
    result = {
        "status": "passed", "archive_sha256": sha256(args.archive),
        "archive_members": members, "manifest_entries_verified": manifest_entries,
        "principal_processes": 588, "supporting_source_processes": 16,
        "derived_csv_json_files_matched": len(originals), "figures_regenerated": 4,
        "reference_profiles_removed_from_analysis_root": True,
        "original_project_and_reference_tables_hidden": args.isolate_analysis,
        "network_namespace_isolation_claimed": False,
    }
    (work / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
