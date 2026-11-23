#!/usr/bin/env python3
"""Execute one permanently claimed block inside an existing Slurm allocation.

Derived from memory_tuner/electronics_experiment.py (run). It never submits,
requeues or retries a job. Differences: two- or four-GPU allocations, the new
launcher, FTA-26-70 roots, all models of a block verified, and E3 restore
periods that resume from a read-only, hash-recorded step checkpoint written
earlier in the same allocation.

`rerun` executes exactly one slot listed in a hash-chained amendment (PLAN.md
section 9) under a new permanent claim, never touching the original attempt.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieee_resub import common as C  # noqa: E402

C.bootstrap()

from memory_tuner.device_contract import Journal, digest, write_new_json  # noqa: E402
from memory_tuner.resolve_hf_snapshot import validate_snapshot  # noqa: E402
from ieee_resub import amendment as AM  # noqa: E402
from ieee_resub import assess as A  # noqa: E402
from ieee_resub.env import checkpoint_dir, launcher_environment, run_name  # noqa: E402

SOURCE = C.REPO_ROOT


# --------------------------------------------------------------------------- runtime
def load_base():
    spec = importlib.util.spec_from_file_location("ieee_resub._matched_runtime",
                                                  SOURCE / "memory_tuner/run_matched_gpu.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def runtime(caps: dict, allocation_gpus: int):
    """run_matched_gpu helpers with the allocation size as a parameter."""
    C.require(allocation_gpus in (2, 4), "Only two- or four-GPU allocations are supported")
    base = load_base()
    base.ALLOCATION_CAP = caps["wall_seconds"]
    base.CLEANUP_RESERVE = 60
    base.IDLE_TIMEOUT = 180
    n = allocation_gpus

    def allocated_identifiers(value):
        assigned = []
        for part in value.split(","):
            interval = re.fullmatch(r"(\d+)-(\d+)", part)
            if interval:
                first, last = map(int, interval.groups())
                C.require(0 <= last - first <= n - 1, "Invalid GPU allocation range")
                assigned.extend(map(str, range(first, last + 1)))
            elif re.fullmatch(r"\d+", part):
                assigned.append(part)
            else:
                assigned.extend(base.parse_uuids(part))
        C.require(len(assigned) == n and len(set(assigned)) == n,
                  f"{n} unambiguous SLURM_JOB_GPUS assignments are required")
        return assigned

    def validate_allocation_inventory(inventory):
        C.require(len(inventory) == n and len({g["uuid"] for g in inventory}) == n,
                  f"{n} distinct allocation GPUs are required")
        C.require(all(g["name"] == "NVIDIA A100-SXM4-40GB" and g["total_mib"] == C.CAPACITY_MIB
                      and g["mig_mode"].lower() == "disabled" for g in inventory),
                  "Allocation must contain full A100-SXM4-40GB GPUs with MIG disabled")

    def subset_ordinals(row):
        indices = [base.strict_int(part, "gpu_subset_ordinals") for part in row["gpu_subset_ordinals"].split(";")]
        count = base.strict_int(row["gpu_count"], "gpu_count")
        C.require(len(indices) == count and indices == sorted(set(indices))
                  and all(i in range(n) for i in indices), "Invalid GPU subset")
        return indices

    def allocation_context():
        job, query, array, task = base.scheduler_job_identity(os.environ)
        assigned = allocated_identifiers(os.environ.get("SLURM_JOB_GPUS", ""))
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        inventory = base.resolve_visible_allocation(base.query_gpus(), visibility)
        result = subprocess.run(["scontrol", "show", "job", query, "--oneliner"],
                                text=True, capture_output=True, check=True, timeout=10)
        fields, record = base.select_scheduler_record(result.stdout, job, array, task)
        C.require(re.search(rf"(?:^|,)gres/gpu={n}(?:,|$)", fields.get("AllocTRES", "")) is not None,
                  f"Scheduler allocation does not reserve exactly {n} GPUs")
        C.require(fields.get("NumNodes") == "1" and fields.get("Requeue") == "0",
                  "One node and Requeue=0 are required")
        C.require(fields.get("Account", "").upper() == C.ACCOUNT, "Allocation is not charged to FTA-26-70")
        start = datetime.fromisoformat(fields["StartTime"]).timestamp()
        end = datetime.fromisoformat(fields["EndTime"]).timestamp()
        C.require(0 < end - start <= base.ALLOCATION_CAP + 1, "Allocation exceeds the frozen wall cap")
        C.require(start <= time.time() < end, "Allocation is not currently active")
        return dict(job_id=job, hostname=socket.gethostname(), gpu_count=n, gpus=inventory,
                    scheduler_gpu_ids=assigned, started_at_epoch=start, ends_at_epoch=end,
                    scheduler_cuda_visible_devices=visibility, scheduler_record=record,
                    scheduler_query_identifier=query, hardware_confinement_claimed=False)

    base.allocated_identifiers = allocated_identifiers
    base.validate_allocation_inventory = validate_allocation_inventory
    base.subset_ordinals = subset_ordinals
    base.allocation_context = allocation_context
    base.validate_identity = functools.partial(A.validate_identity, allocation_gpus=n, is_idle=base.is_idle)
    return base


# --------------------------------------------------------------------------- protocol
def load(protocol_path: Path, expected_hash: str, study: str, amendment_path: Path | None = None,
         amendment_hash: str | None = None):
    """Frozen protocol and matrix; with an amendment, the executing source must be
    the amended snapshot recorded in it (hash-chained to the protocol)."""
    C.require(re.fullmatch("[a-f0-9]{64}", expected_hash or "") is not None, "Invalid protocol anchor")
    C.require(digest(protocol_path) == expected_hash, "Protocol changed")
    protocol = json.loads(protocol_path.read_text())
    C.require(protocol["study"] == study and protocol["status"] == "frozen", "Protocol is not a frozen study protocol")
    matrix = C.EXPERIMENT_DIR / "matrices" / f"{study}_matrix.csv"
    C.require(digest(matrix) == protocol["matrix_sha256"], "Matrix changed")
    amendment = None
    if amendment_path is not None:
        amendment = AM.load(amendment_path, amendment_hash, protocol_path, protocol, expected_hash, study)
    from ieee_resub.plan import source_hashes
    expected_sources = amendment["source_files_sha256"] if amendment else protocol["source_files_sha256"]
    C.require(expected_sources == source_hashes(), "Executing source differs from the frozen protocol"
              + (" and its amendment" if amendment else ""))
    rows = C.read_rows(matrix)
    C.require(len(rows) == protocol["planned_invocations"], "Changed slot inventory")
    C.require(len({r["experiment_id"] for r in rows}) == len(rows), "Duplicate invocation")
    for row in rows:
        C.require(C.SAFE_ID.fullmatch(row["experiment_id"]) and C.SAFE_ID.fullmatch(row["block_id"]), "Unsafe id")
        C.require(row["run_group"] == protocol["group"], "Group mismatch")
    if amendment is not None:
        AM.check_against_matrix(amendment, rows)
    return protocol, rows, amendment


def clean_source(expected: str) -> str:
    C.require(re.fullmatch(r"[a-f0-9]{40}", expected or "") is not None, "40-hex source commit required")
    head = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain",
                                     "--untracked-files=normal"], text=True)
    C.require(head == expected and not dirty, "A clean frozen source commit is required")
    return head


def verify_inputs(protocol, selected):
    for relative, expected in protocol["data_sha256"].items():
        if any(relative.startswith("data/" + C.DATA_SUBDIRS[r["dataset"]] + "/") for r in selected):
            C.require(digest(C.CLUSTER_PROJECT_ROOT / relative) == expected, "Frozen dataset changed: " + relative)
    snapshots = {}
    for model in sorted({r["model"] for r in selected}):
        cache = protocol["model_cache"][model]
        snapshot = validate_snapshot(Path(cache["path"]))
        for name, expected in cache["files_sha256"].items():
            C.require(digest(snapshot / name) == expected, f"Frozen model changed: {model}/{name}")
        snapshots[model] = snapshot
    return snapshots


# --------------------------------------------------------------------------- restore
def checkpoint_manifest(folder: Path) -> dict:
    files = {}
    for path in sorted(folder.rglob("*")):
        C.require(not path.is_symlink(), "Linked file in checkpoint")
        if path.is_file():
            stat = path.stat()
            files[str(path.relative_to(folder))] = dict(sha256=digest(path), size_bytes=stat.st_size,
                                                        mtime_ns=stat.st_mtime_ns)
    return files


def prepare_restore_source(row, selected, results, job_id, gpu_count) -> Path:
    """Locate, validate and freeze the earlier long run's step checkpoint."""
    source = next(r for r in selected if r["experiment_id"] == row["resume_source_experiment_id"])
    C.require(int(source["period"]) < int(row["period"]), "Restore source must run earlier in the block")
    C.require(any(r["experiment_id"] == source["experiment_id"] and r.get("launched") for r in results),
              "Restore source was not launched")
    folder = checkpoint_dir(source, job_id) / f"global_step_{int(row['resume_step'])}"
    validate_checkpoint_layout(folder, gpu_count)
    for path in [folder, *folder.rglob("*")]:
        path.chmod(path.stat().st_mode & ~0o222)
    return folder


def validate_checkpoint_layout(folder: Path, gpu_count: int) -> None:
    C.require(folder.is_dir() and (folder / "data.pt").is_file(), "Source checkpoint is absent")
    for rank in range(gpu_count):
        for kind in ("model", "optim", "extra_state"):
            C.require((folder / "actor" / f"{kind}_world_size_{gpu_count}_rank_{rank}.pt").is_file(),
                      f"Source checkpoint lacks {kind} shard of rank {rank}")


def original_manifest(entry: dict, group: Path) -> dict:
    """The restore-source manifest that the original attempt recorded before its run."""
    source = entry["restore_source"]
    record = Path(source["manifest_record"])
    C.require(record == group / entry["experiment_id"] / f"attempt-{entry['original_job_id']}" / "restore-source.json",
              "Manifest record is not the original attempt's")
    C.require(digest(record) == source["manifest_record_sha256"], "Original manifest record changed")
    original = json.loads(record.read_text())
    C.require(original["folder"] == source["original_folder"] and len(original["manifest"]) == source["manifest_files"],
              "Original manifest record differs from the amendment")
    return original["manifest"]


def prepare_preserved_source(entry: dict, row: dict, group: Path, gpu_count: int):
    """Amendment rerun: the preserved copy of the original step checkpoint must be
    file-by-file identical (SHA-256, size, mtime) to the original manifest."""
    folder = Path(entry["restore_source"]["folder"])
    preserved = (C.CLUSTER_SCRATCH_ROOT / "preserved-checkpoints").resolve()
    C.require(folder.resolve().is_relative_to(preserved) and not folder.is_symlink()
              and folder.name == f"global_step_{int(row['resume_step'])}",
              "Restore source is not a preserved step checkpoint")
    validate_checkpoint_layout(folder, gpu_count)
    expected = original_manifest(entry, group)
    for path in [folder, *folder.rglob("*")]:
        path.chmod(path.stat().st_mode & ~0o222)
    manifest = checkpoint_manifest(folder)
    C.require(manifest == expected, "Preserved checkpoint differs from the original manifest")
    return folder, manifest, expected


def original_slot(entry: dict, group: Path) -> dict:
    """The original attempt must be terminal, unresolved, and is never modified."""
    block_dir = group / "blocks" / entry["block_id"]
    block = json.loads((block_dir / "block.json").read_text())
    C.require(str(block["allocation"]["job_id"]) == entry["original_job_id"], "Original block job differs")
    receipt_path = block_dir / f"period-{int(entry['period'])}.json"
    receipt = json.loads(receipt_path.read_text())
    C.require(receipt["experiment_id"] == entry["experiment_id"] and receipt.get("launched") is True,
              "Original receipt differs")
    C.require(receipt["status"] == entry["required_original_status"],
              "Original slot is not unresolved; diagnosed or completed outcomes are never rerun")
    C.require(receipt.get("attempt_path") == entry["original_attempt"], "Original attempt path differs")
    C.require((block_dir / "summary.json").is_file(), "Original block has not finished")
    files = {p.name: digest(p) for p in sorted(block_dir.iterdir()) if p.is_file() and p.name != "allocation-trace.jsonl"}
    return dict(block_dir=str(block_dir), receipt=str(receipt_path), receipt_status=receipt["status"],
                receipt_validation_error=receipt.get("validation_error") or receipt.get("reason"),
                files_sha256=files)


def release_checkpoints(selected, job, block_dir):
    """Frozen retention policy: model checkpoints are deleted at the end of the block,
    after every invocation's checkpoint-inventory.json and the restore-source hash
    manifest were written. Only this block's own scratch directories are touched."""
    import shutil
    released = []
    root = (C.CLUSTER_CHECKPOINT_ROOT).resolve()
    for row in selected:
        if row["checkpoint_to_scratch"] != "True":
            continue
        folder = checkpoint_dir(row, job)
        C.require(folder.resolve().is_relative_to(root) and folder.name == f"attempt-{job}",
                  "Refusing to delete outside this block's checkpoint directories")
        if folder.exists():
            for path in [folder, *folder.rglob("*")]:
                if path.is_dir() and not path.is_symlink():
                    path.chmod(path.stat().st_mode | 0o700)
            shutil.rmtree(folder)
            released.append(str(folder))
    write_new_json(block_dir / "checkpoint-release.json", dict(released=released, policy=(
        "delete after inventory; step checkpoints are working state, not evidence")))


# --------------------------------------------------------------------------- execution
def run(args):
    """One frozen block (array index). With an amendment, only blocks that it lists
    as never launched may start, and they execute the amended snapshot."""
    protocol, rows, amendment = load(args.protocol, args.protocol_sha256, args.study,
                                     args.amendment, args.amendment_sha256)
    selected = sorted((r for r in rows if int(r["block_index"]) == args.index), key=lambda r: int(r["period"]))
    C.require(selected and len({r["block_id"] for r in selected}) == 1, "Unknown or ambiguous block index")
    C.require([int(r["period"]) for r in selected] == list(range(1, len(selected) + 1)), "Missing/duplicate periods")
    claim = dict()
    if amendment is not None:
        launch = amendment["amended_launch_blocks"]
        C.require(args.index in launch["array_indices"]
                  and launch["block_ids"][launch["array_indices"].index(args.index)] == selected[0]["block_id"],
                  "The amendment does not list this block as never launched")
        claim = dict(amendment_id=amendment["amendment_id"], amendment_sha256=args.amendment_sha256,
                     claim_kind="first_launch_under_amendment")
    block_dir = C.CLUSTER_OUTPUT_ROOT / protocol["group"] / "blocks" / selected[0]["block_id"]
    return execute(args, protocol, selected, block_dir, claim, args.index)


def rerun(args):
    """Amendment rerun of exactly one listed slot (array index into its reruns):
    a new permanent claim blocks/<block>-amendment-NN, the frozen matrix row
    unchanged (seed, settings, timeout), the original attempt untouched, and for
    restore slots the preserved step checkpoint verified against the original
    manifest before and after the invocation."""
    C.require(args.amendment is not None, "An amendment rerun needs its amendment")
    protocol, rows, amendment = load(args.protocol, args.protocol_sha256, args.study,
                                     args.amendment, args.amendment_sha256)
    C.require(0 <= args.index < len(amendment["reruns"]), "Unknown amendment rerun index")
    entry = amendment["reruns"][args.index]
    row = AM.matrix_row(rows, entry["experiment_id"])
    group = C.CLUSTER_OUTPUT_ROOT / protocol["group"]
    original = original_slot(entry, group)
    if row["role"] == "restore":
        # Verified before the claim, so that a changed copy never consumes it.
        prepare_preserved_source(entry, row, group, int(row["gpu_count"]))
    else:
        C.require("restore_source" not in entry, "Only restore slots resume from a checkpoint")
    claim = dict(amendment_id=amendment["amendment_id"], amendment_sha256=args.amendment_sha256,
                 claim_kind="amendment_rerun", rerun=entry, original=original)
    return execute(args, protocol, [row], group / entry["claim"], claim, args.index, rerun_entry=entry)


def execute(args, protocol, selected, block_dir, claim, array_index, rerun_entry=None):
    from memory_tuner.run_offload_factorial import InterruptState, enable_subreaper, inventory
    from memory_tuner.host_memory_monitor import HostMemorySampler

    group = C.CLUSTER_OUTPUT_ROOT / protocol["group"]
    base = runtime(protocol["caps"], protocol["allocation_gpus"])
    ledger = base.claim_pair(block_dir)  # mkdir is the permanent, job-independent claim
    trace = Journal(block_dir / "allocation-trace.jsonl")
    interrupt = InterruptState()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt.signal)
    allocation, results, fatal = None, [], None
    try:
        head = clean_source(args.source_commit)
        C.require(os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Requeued job forbidden")
        allocation = base.allocation_context()
        base.validate_allocation_inventory(allocation["gpus"])
        C.require(os.environ.get("SLURM_ARRAY_TASK_ID") == str(array_index), "Array slot differs")
        ledger.emit("block_claimed", source_commit=head, protocol_sha256=args.protocol_sha256,
                    block_id=selected[0]["block_id"], rows=selected, allocation=allocation, **claim)
        write_new_json(block_dir / "block.json", dict(source_commit=head, protocol=protocol,
                       protocol_sha256=args.protocol_sha256, allocation=allocation, rows=selected, **claim))
        snapshots = verify_inputs(protocol, selected)
        allocated = [g["uuid"] for g in allocation["gpus"]]
        deadline = time.monotonic() + allocation["ends_at_epoch"] - time.time() - 10
        enable_subreaper()
        launcher = SOURCE / C.LAUNCHER
        for row in selected:
            period = int(row["period"])
            result = dict(experiment_id=row["experiment_id"], status="not_launched", launched=False)
            attempt = None
            try:
                C.require(fatal is None and not interrupt.requested, fatal or "Interrupted")
                base.PAYLOAD_TIMEOUT = int(row["per_process_timeout_seconds"])
                C.require(deadline - time.monotonic() >= base.PAYLOAD_TIMEOUT + 60, "Insufficient remaining allocation")
                job = allocation["job_id"]
                resume_from = None
                if row["role"] == "restore":
                    try:
                        if rerun_entry is None:
                            resume_from = prepare_restore_source(row, selected, results, job, int(row["gpu_count"]))
                        else:  # verified against the original manifest before the claim (rerun())
                            resume_from = Path(rerun_entry["restore_source"]["folder"])
                            validate_checkpoint_layout(resume_from, int(row["gpu_count"]))
                    except (ValueError, OSError, StopIteration) as error:
                        # Not fatal: the slot stays not_launched with its reason.
                        result.update(reason=f"Restore precondition failed: {error}")
                        continue
                C.require(interrupt.call(base.wait_idle, allocated, trace, period, "pre_idle", deadline),
                          "Allocation is not idle")
                expected = [allocated[i] for i in base.subset_ordinals(row)]
                invocation = row["experiment_id"] + "-j" + job
                attempt = C.CLUSTER_OUTPUT_ROOT / run_name(row, job)
                if rerun_entry is None:
                    C.require(not attempt.parent.exists(), "An invocation may not be replaced")
                else:
                    original_attempt = Path(rerun_entry["original_attempt"])
                    C.require(original_attempt.is_dir() and original_attempt.parent == attempt.parent
                              and not attempt.exists(), "An amendment rerun adds an attempt next to the original")
                attempt.mkdir(parents=True)
                if resume_from is not None:
                    record = dict(folder=str(resume_from), source_experiment_id=row["resume_source_experiment_id"],
                                  manifest=checkpoint_manifest(resume_from))
                    if rerun_entry is not None:
                        record.update(amendment_id=claim["amendment_id"],
                                      original_manifest_record=rerun_entry["restore_source"]["manifest_record"],
                                      matches_original_manifest=record["manifest"] == original_manifest(rerun_entry, group))
                    write_new_json(attempt / "restore-source.json", record)
                    C.require(record.get("matches_original_manifest", True), "Restore source differs from its manifest")
                values = launcher_environment(
                    row, job_id=job, source_root=SOURCE, output_root=C.CLUSTER_OUTPUT_ROOT, attempt=attempt,
                    invocation=invocation, period=period, task_uuids=expected, allocation_uuids=allocated,
                    snapshot=snapshots[row["model"]], resume_from=resume_from)
                write_new_json(attempt / "dispatch.json", dict(row=row, environment=values,
                               protocol_sha256=args.protocol_sha256, source_commit=head, invocation_id=invocation))
                ledger.emit("dispatched", period=period, experiment_id=row["experiment_id"], invocation_id=invocation)
                started_ns = time.time_ns()
                with HostMemorySampler(attempt, job):
                    execution = interrupt.payload(base.run_payload, ["bash", str(launcher)],
                                                  base.clean_environment(values), attempt / "launcher.log",
                                                  allocated, expected, ledger, trace, period, deadline)
                execution.update(started_ns=started_ns, finished_ns=time.time_ns())
                write_new_json(attempt / "execution.json", execution)
                if row["checkpoint_to_scratch"] == "True":
                    write_new_json(attempt / "checkpoint-inventory.json",
                                   A.checkpoint_inventory(checkpoint_dir(row, job)))
                if resume_from is not None:
                    post = dict(folder=str(resume_from), manifest=checkpoint_manifest(resume_from))
                    if rerun_entry is not None:
                        post["matches_original_manifest"] = post["manifest"] == original_manifest(rerun_entry, group)
                    write_new_json(attempt / "restore-source-post.json", post)
                result.update(launched=execution["spawned"], attempt_path=str(attempt),
                              invocation_gpu_seconds=execution["invocation_elapsed_seconds"] * int(row["gpu_count"]))
                clean = interrupt.call(base.wait_idle, allocated, trace, period, "post_idle", deadline,
                                       allow_pending=True)
                result.update(A.assess(attempt, row, execution, allocation, protocol, head, base))
                if not clean or not execution["cleanup"]["complete"]:
                    fatal = "Post-invocation cleanup/idle gate failed"
                    result.update(status="unresolved", validation_error=fatal)
            except (Exception, KeyboardInterrupt) as error:
                fatal = f"{type(error).__name__}: {error}"
                result.update(reason=fatal)
            finally:
                result["evidence_files"] = inventory(attempt) if attempt else {}
                write_new_json(block_dir / f"period-{period}.json", result)
                ledger.emit("terminal", period=period, **result)
                results.append(result)
        write_new_json(block_dir / "summary.json", dict(results=results, fatal=fatal))
        if protocol["caps"].get("checkpoint_retention") == "delete_at_block_end":
            release_checkpoints(selected, allocation["job_id"], block_dir)
    except (Exception, KeyboardInterrupt) as error:
        write_new_json(block_dir / "setup-error.json", dict(error=f"{type(error).__name__}: {error}",
                       source_commit=args.source_commit, allocation=allocation))
        raise
    finally:
        ledger.close()
        trace.close()
    return int(any(r["status"] not in ("within_margin", "above_margin", "memory_failure") for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "rerun"):
        command = sub.add_parser(name, help="frozen block (array index)" if name == "run"
                                 else "amendment rerun (array index into the amendment's reruns)")
        command.add_argument("--study", choices=C.ALL_STUDIES, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--protocol-sha256", required=True)
        command.add_argument("--source-commit", required=True)
        command.add_argument("--index", type=int, required=True)
        command.add_argument("--amendment", type=Path, required=name == "rerun")
        command.add_argument("--amendment-sha256", required=name == "rerun")
    args = parser.parse_args()
    C.require((args.amendment is None) == (args.amendment_sha256 is None), "Amendment path and hash go together")
    return run(args) if args.command == "run" else rerun(args)


if __name__ == "__main__":
    raise SystemExit(main())
