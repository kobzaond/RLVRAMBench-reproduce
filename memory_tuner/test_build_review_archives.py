from pathlib import Path
import json
import shutil

import pytest

from memory_tuner.build_review_archives import safe_member
from memory_tuner import build_review_archives as module


def test_archive_members_cannot_escape_or_use_absolute_links(tmp_path):
    member = tmp_path / "trial.json"
    member.write_text("{}")
    relative = tmp_path / "relative.json"
    relative.symlink_to("trial.json")
    absolute = tmp_path / "absolute.json"
    absolute.symlink_to(member)
    assert safe_member(tmp_path, "relative.json") == relative
    for name in ("../trial.json", str(member), "absolute.json", "missing.json"):
        with pytest.raises(ValueError):
            safe_member(tmp_path, name)


def test_direct_tex_handoff_replaces_stale_author_boilerplate(tmp_path, monkeypatch):
    if shutil.which("zstd") is None:
        pytest.skip("archive compressor unavailable")
    root = tmp_path / "source"
    directory = root / "paper/submission/ieee_access_submission"
    directory.mkdir(parents=True)
    verification = {"source_bundle_verified_after_extraction": True,
                    "author_blockers_remain": ["corresponding_author_email", "artifact: doi"]}
    for name, field in (
        ("RLVRAMBench_IEEE_Access_submission.tex", "tex_sha256"),
        ("RLVRAMBench_IEEE_Access_submission.pdf", "pdf_sha256"),
        ("RLVRAMBench_supplement.tex", "supplement_tex_sha256"),
        ("RLVRAMBench_supplement.pdf", "supplement_pdf_sha256"),
    ):
        path = directory / name
        path.write_text("Authoritative submission manuscript. Edit this TeX directly")
        verification[field] = module.file_sha256(path)
    bundle = directory.parent / "RLVRAMBench_IEEE_Access_source.zip"
    bundle.write_bytes(b"source-bundle-fixture")
    verification["source_bundle_sha256"] = module.file_sha256(bundle)
    (directory / "export_verification.json").write_text(json.dumps(verification))
    (root / module.MANIFEST).parent.mkdir(parents=True)
    (root / module.MANIFEST).write_text("path\n")
    (root / "paper/artifact_manifest.md").write_text("fixture")
    monkeypatch.setattr(module, "audit_package", lambda _: ([], ["stale license choice"], {}))
    monkeypatch.setattr(module, "verify_manifest", lambda *_: [])
    monkeypatch.setattr(module, "read_csv", lambda _: [])
    result = module.build(root, tmp_path / "release", "direct")
    assert result["author_supplied_upload_blockers"] == ["corresponding_author_email"]
    assert result["submission_members"] == 3
