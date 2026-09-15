#!/usr/bin/env python3
"""Resolve and validate a model snapshot from the project-local HF cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty cached model file: {path}")


def validate_snapshot(snapshot: Path) -> Path:
    snapshot = snapshot.resolve()
    _require_file(snapshot / "config.json")
    _require_file(snapshot / "tokenizer_config.json")
    if not (snapshot / "tokenizer.json").is_file():
        _require_file(snapshot / "vocab.json")
        _require_file(snapshot / "merges.txt")

    index_path = snapshot / "model.safetensors.index.json"
    single_weight = snapshot / "model.safetensors"
    if index_path.is_file():
        _require_file(index_path)
        index = json.loads(index_path.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"empty safetensors weight map: {index_path}")
        for shard in shards:
            _require_file(snapshot / shard)
    else:
        _require_file(single_weight)
    return snapshot


def resolve_snapshot(hf_home: Path, model: str) -> Path:
    direct = Path(model)
    if direct.is_dir():
        return validate_snapshot(direct)

    repo = hf_home / "hub" / f"models--{model.replace('/', '--')}"
    snapshots_root = repo / "snapshots"
    ref = repo / "refs" / "main"
    candidates: list[Path] = []
    if ref.is_file():
        revision = ref.read_text().strip()
        if revision:
            candidates.append(snapshots_root / revision)
    if snapshots_root.is_dir():
        candidates.extend(
            path
            for path in sorted(snapshots_root.iterdir(), reverse=True)
            if path.is_dir()
        )

    errors = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            return validate_snapshot(candidate)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{candidate}: {error}")
    detail = "; ".join(errors) if errors else "no cached snapshots found"
    raise ValueError(f"no complete project-local snapshot for {model}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(resolve_snapshot(args.hf_home.resolve(), args.model))


if __name__ == "__main__":
    main()
