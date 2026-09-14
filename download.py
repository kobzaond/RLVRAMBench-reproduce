#!/usr/bin/env python3
"""Download the exact Hugging Face evidence revision and verify its SHA-256."""
import argparse
import json
import shutil
from pathlib import Path

from reproduce import ROOT, sha256


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("downloads"))
    args = parser.parse_args()
    release = json.loads((ROOT / "release.json").read_text())
    revision = release["huggingface_revision"]
    if not revision or len(revision) != 40:
        raise SystemExit("release.json does not identify a verified Hugging Face commit")
    target = args.output_dir / Path(release["archive"]["path"]).name
    if target.exists():
        if sha256(target) != release["archive"]["sha256"]:
            raise SystemExit("Existing download differs; refusing to overwrite it")
    else:
        downloaded = Path(hf_hub_download(
            repo_id=release["huggingface_repository"], repo_type="dataset",
            revision=revision, filename=release["archive"]["path"],
            cache_dir=args.output_dir / ".hf-cache",
        ))
        if sha256(downloaded) != release["archive"]["sha256"]:
            raise SystemExit("Downloaded archive SHA-256 differs from the release lock")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, target)
    print(target)


if __name__ == "__main__":
    main()
