"""Package validated scientific evidence without overwriting earlier releases."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

from memory_tuner.artifact_manifest import (
    file_sha256, publication_empty_directories, read_csv, verify_manifest,
)
from memory_tuner.standard_grpo_readiness import audit_package


MANIFEST = Path("profiles/benchmark/publication-artifact-manifest.csv")


def safe_member(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"archive member must be artifact-relative: {name}")
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing member or external symlink: {name}")
    if path.is_symlink() and Path(path.readlink()).is_absolute():
        raise ValueError(f"nonportable absolute symlink: {name}")
    return path


def build(root: Path, destination: Path, tag: str) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", tag):
        raise ValueError("release tag must be a simple filename component")
    errors, author_blockers, _ = audit_package(root)
    errors += verify_manifest(root, root / MANIFEST)
    if errors:
        raise ValueError("release validation failed:\n" + "\n".join(errors))
    names = sorted({row["path"] for row in read_csv(root / MANIFEST)} | {
        str(MANIFEST), "paper/artifact_manifest.md",
    })
    members = [(name, safe_member(root, name)) for name in names]
    empty_directories = publication_empty_directories(root)
    destination.mkdir(parents=True, exist_ok=True)
    review = destination / f"RLVRAMBench-review-artifact-{tag}.tar.zst"
    submission = destination / f"RLVRAMBench-submission-scientific-package-{tag}.zip"
    metadata = destination / f"RLVRAMBench-release-{tag}.json"
    for path in (review, submission, metadata):
        if path.exists():
            raise FileExistsError(f"preserving existing release: {path}")
    with review.open("xb") as output:
        compressor = subprocess.Popen(["zstd", "-T2", "-8", "-c"],
                                      stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=compressor.stdin, mode="w|") as archive:
                for path in empty_directories:
                    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError(f"Unsafe empty evidence directory: {path}")
                    archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
                for name, path in members:
                    archive.add(path, arcname=name, recursive=False)
        finally:
            compressor.stdin.close()
            status = compressor.wait()
        if status:
            raise RuntimeError(f"zstd exited with {status}; {review} is incomplete")

    tex_directory = root / "paper/submission/ieee_access_submission"
    stem = "RLVRAMBench_IEEE_Access_submission"
    authoritative = tex_directory / f"{stem}.tex"
    if authoritative.is_file() and (
            "Authoritative submission manuscript. Edit this TeX directly"
            in authoritative.read_text()):
        verification = json.loads((tex_directory / "export_verification.json").read_text())
        for name, field in (
            (f"{stem}.tex", "tex_sha256"), (f"{stem}.pdf", "pdf_sha256"),
            ("RLVRAMBench_supplement.tex", "supplement_tex_sha256"),
            ("RLVRAMBench_supplement.pdf", "supplement_pdf_sha256"),
        ):
            if file_sha256(tex_directory / name) != verification[field]:
                raise ValueError(f"Direct submission changed since verification: {name}")
        source_zip = root / "paper/submission/RLVRAMBench_IEEE_Access_source.zip"
        if file_sha256(source_zip) != verification["source_bundle_sha256"]:
            raise ValueError("Direct TeX source bundle changed since verification")
        if not verification["source_bundle_verified_after_extraction"]:
            raise ValueError("Direct TeX source bundle has not passed an extracted rebuild")
        # The historical audit checks the archived cohort, not current author
        # decisions. Use the direct-TeX handoff instead of stale JSS boilerplate.
        # A DOI is optional deposition work, not a manuscript-upload prerequisite.
        author_blockers = [
            item for item in verification["author_blockers_remain"]
            if item != "artifact: doi"
        ]
        submission_names = {
            str((tex_directory / f"{stem}.pdf").relative_to(root)),
            str((tex_directory / "RLVRAMBench_supplement.pdf").relative_to(root)),
            str(source_zip.relative_to(root)),
        }
    else:
        submission_names = {
        "paper/phaseguard_paper_draft.md", "paper/references.bib",
        "paper/claim_ledger.md", "paper/claim_evidence_matrix.md",
        "paper/standard_grpo_evidence.md", "paper/submission_readiness.md",
        "paper/scientific_revision_protocol_2026-09-13.md",
        "paper/scientific_revision_reproduction_audit_2026-09-13.md",
        "paper/scientific_revision_summary_2026-09-13.md",
        "paper/scientific_revision_build_audit_2026-09-13.md",
        "paper/scientific_revision_lifecycle_audit_2026-09-13.md",
        "paper/scientific_revision_dataset_attribution_2026-09-13.md",
        "paper/scientific_revision_matrix_hashes_2026-09-13.json",
        "paper/REPRODUCING_RLVRAMBench.md", "profiles/standard_grpo/summary.json",
        }
        submission_names.update(str(p.relative_to(root)) for p in
                                (root / "paper/submission").iterdir() if p.is_file())
        submission_names.update(str(p.relative_to(root)) for p in
                                (root / "paper/figures").glob("standard_grpo_*")
                                if p.suffix in (".png", ".pdf"))
    with zipfile.ZipFile(submission, mode="x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(submission_names):
            archive.write(safe_member(root, name), arcname=name)
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"submission ZIP integrity error: {bad_member}")
    result = {
        "tag": tag, "review_members": len(names) + len(empty_directories),
        "empty_evidence_directories": [
            str(path.relative_to(root)) for path in empty_directories
        ],
        "manifest_entries": len(read_csv(root / MANIFEST)),
        "submission_members": len(submission_names),
        "author_supplied_upload_blockers": author_blockers,
        "archives": [
            {"path": str(p), "size_bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in (review, submission)
        ],
    }
    with metadata.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.destination, args.tag), indent=2))


if __name__ == "__main__":
    main()
