#!/usr/bin/env python3
"""Capture reproducibility metadata for one colocated RL trial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS",
    "CUDA_VISIBLE_DEVICES",
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(command: list[str], timeout: int = 120) -> dict:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as error:  # Metadata capture must not hide trial output.
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
        }


def resolve_model_revision(hf_home: Path, model: str) -> dict:
    repo = hf_home / "hub" / f"models--{model.replace('/', '--')}"
    ref = repo / "refs" / "main"
    snapshots = repo / "snapshots"
    available = (
        sorted(path.name for path in snapshots.iterdir() if path.is_dir())
        if snapshots.is_dir()
        else []
    )
    revision = ref.read_text().strip() if ref.is_file() else ""
    if not revision and len(available) == 1:
        revision = available[0]
    return {
        "requested_model": model,
        "resolved_revision": revision,
        "available_snapshots": available,
        "cache_repo": str(repo),
    }


def data_metadata(data_dir: Path) -> dict:
    result = {}
    for name in ("train.parquet", "test.parquet", "profile.json"):
        path = data_dir / name
        if path.is_file():
            result[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return result


def repository_metadata(repo_root: Path) -> dict:
    head = command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = command(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ]
    )
    status_text = status["stdout"]
    return {
        "head": head["stdout"] if head["returncode"] == 0 else "",
        "head_error": head["stderr"] if head["returncode"] != 0 else "",
        "dirty": bool(status_text),
        "status_entry_count": len(status_text.splitlines()),
        "status_sha256": hashlib.sha256(status_text.encode()).hexdigest(),
    }


def container_software(
    image: Path,
    runtime_home: Path,
    verl_root: Path,
    gpu_uuids: list[str] | None = None,
    repo_root: Path | None = None,
) -> dict:
    script = r"""
import importlib.metadata
import json
import platform
import torch
__DEVICE_IMPORT__

packages = {}
for name in (
    "torch",
    "vllm",
    "transformers",
    "ray",
    "flash-attn",
    "tensordict",
    "numpy",
    "accelerate",
):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
try:
    nccl_version = ".".join(str(item) for item in torch.cuda.nccl.version())
except Exception:
    nccl_version = None
print(json.dumps({
    "python": platform.python_version(),
    "packages": packages,
    "torch_cuda": torch.version.cuda,
    "torch_git_version": getattr(torch.version, "git_version", None),
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "nccl": nccl_version,
    __DEVICE_FIELD__
}, sort_keys=True))
""".strip()
    extra = []
    if gpu_uuids is not None:
        extra = ["--env", "CUDA_VISIBLE_DEVICES=" + ",".join(gpu_uuids),
                 "--env", "RLVRAM_GPU_UUIDS=" + ",".join(gpu_uuids),
                 "--env", f"PYTHONPATH={repo_root}",
                 "--bind", f"{repo_root}:{repo_root}"]
        script = script.replace(
            "__DEVICE_IMPORT__",
            "from memory_tuner.device_contract import check_cuda, requested_uuids")
        script = script.replace(
            "__DEVICE_FIELD__", '"cuda_devices": check_cuda(requested_uuids()),')
    else:
        script = script.replace("__DEVICE_IMPORT__", "").replace("__DEVICE_FIELD__", "")
    result = command(
        [
            "apptainer",
            "exec",
            "--nv",
            "--cleanenv",
            "--home",
            str(runtime_home),
            "--bind",
            f"{verl_root}:{verl_root}",
            *extra,
            str(image),
            "python",
            "-c",
            script,
        ]
    )
    if result["returncode"] == 0:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
    return {
        "capture_error": result["stderr"] or result["stdout"],
        "returncode": result["returncode"],
    }


def build_metadata(args: argparse.Namespace) -> dict:
    gpu_uuids = None
    if "RLVRAM_GPU_UUIDS" in os.environ:
        if __package__:
            from .device_contract import requested_uuids
        else:
            from device_contract import requested_uuids
        gpu_uuids = requested_uuids()
    image = args.image.resolve()
    image_stat = image.stat()
    inspect = command(["apptainer", "inspect", "--json", str(image)])
    try:
        inspect_json = json.loads(inspect["stdout"])
    except json.JSONDecodeError:
        inspect_json = {
            "capture_error": inspect["stderr"] or inspect["stdout"],
            "returncode": inspect["returncode"],
        }
    gpu_result = command(
        [
            "nvidia-smi",
            *(["--id=" + ",".join(gpu_uuids)] if gpu_uuids is not None else []),
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    software = (container_software(
        image, args.runtime_home.resolve(), args.verl_root.resolve(),
        gpu_uuids, args.repo_root.resolve()) if gpu_uuids is not None else
        container_software(image, args.runtime_home.resolve(), args.verl_root.resolve()))
    if gpu_uuids is not None:
        observed = list(csv.reader(gpu_result["stdout"].splitlines(), skipinitialspace=True))
        if (gpu_result["returncode"] != 0 or "capture_error" in software
                or len(observed) != len(gpu_uuids)
                or {row[2].strip() for row in observed if len(row) == 5} != set(gpu_uuids)):
            raise ValueError("Required task environment/device capture failed")
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
        },
        "slurm": {key: os.environ.get(key, "") for key in SLURM_KEYS},
        "gpus_csv": gpu_result["stdout"],
        "gpus_capture_error": (
            gpu_result["stderr"] if gpu_result["returncode"] != 0 else ""
        ),
        "container": {
            "path": str(image),
            "sha256": args.image_sha256,
            "size_bytes": image_stat.st_size,
            "mtime_ns": image_stat.st_mtime_ns,
            "inspect": inspect_json,
        },
        "software": software,
        "repository": repository_metadata(args.repo_root.resolve()),
        "model": resolve_model_revision(args.hf_home.resolve(), args.model),
        "dataset": (
            data_metadata(args.data_dir.resolve())
            if args.data_dir is not None
            else {}
        ),
    }
    if gpu_uuids is not None:
        result["gpu_device_contract"] = {
            "requested_uuids": gpu_uuids, "task_gpu_count": len(gpu_uuids),
            "invocation_id": os.environ["RLVRAM_INVOCATION_ID"],
            "hardware_confinement_claimed": False,
        }
        resolved = Path(args.resolved_model_path).resolve()
        if not resolved.is_dir():
            raise ValueError("Resolved model snapshot is absent")
        result["model"]["actual_snapshot_path"] = str(resolved)
        result["model"]["resolved_revision"] = resolved.name
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--runtime-home", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--resolved-model-path", type=Path)
    args = parser.parse_args()
    metadata = build_metadata(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if "RLVRAM_GPU_UUIDS" in os.environ:
        if __package__:
            from .device_contract import write_new_json
        else:
            from device_contract import write_new_json
        write_new_json(args.output, metadata)
    else:
        args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote environment metadata to {args.output}")


if __name__ == "__main__":
    main()
