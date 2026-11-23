#!/usr/bin/env python3
"""Download the five release archives at one Hugging Face commit and verify their SHA-256.

Optional helper; reproduce_all.py can download the same files itself with the
standard library. Uses release.json (huggingface_revision) unless --revision is given.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("downloads"))
    parser.add_argument("--revision", help="40-hex dataset commit; defaults to release.json")
    args = parser.parse_args()
    release = json.loads((ROOT / "release.json").read_text())
    revision = args.revision or release["huggingface_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise SystemExit("No verified Hugging Face commit: pass --revision or fill release.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in release["release_archives"]:
        target = args.output_dir / Path(entry["path"]).name
        if target.exists():
            if sha256(target) != entry["sha256"]:
                raise SystemExit(f"Existing {target} differs; refusing to overwrite it")
        else:
            downloaded = Path(hf_hub_download(
                repo_id=release["huggingface_repository"], repo_type="dataset",
                revision=revision, filename=entry["path"], cache_dir=args.output_dir / ".hf-cache",
            ))
            if sha256(downloaded) != entry["sha256"]:
                raise SystemExit(f"Downloaded {entry['path']} differs from the release lock")
            shutil.copyfile(downloaded, target)
        print(target)


if __name__ == "__main__":
    main()
