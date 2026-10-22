"""Opt-in, restart-safe first-block gate; never retries an ambiguous submission.

Run from the mutable clone with the project's .venv-reproduce Python.
Default/--once only queries accounting and prints status. --execute writes only
the fixed project-local state directory and may submit indices 1-17%3 once.
An incomplete transaction or failed gate requires human review, not deletion of
the intent followed by a retry. No cancellation, requeue, refit or source edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from memory_tuner.capture_allocation_accounting import FIELDS, TERMINAL_STATES
from memory_tuner.device_contract import digest, sync_directory, write_new_json

SOURCE = Path("/mnt/proj3/open-35-44/verl/experiments/offload-factorial-3d6ecac")
COMMIT = "3d6ecac1852e16a0422c9c72a230187be663f8b5"
FREEZE = "5df43f35e0b02c23e4816b7e860c5a613176da41f5adb27e57f79ac2becf1fb9"
PROJECT = Path("/mnt/proj3/open-35-44/verl/rl")
FIRST = "5069460"
STATE = PROJECT / "output/offload-factorial-watch-5069460"
RAW = Path("output/offload_factorial")
RESULTS = {"slots.csv", "configurations.csv", "prediction_scores.csv",
           "offload_pairs.csv", "block_contrasts.csv", "summary.json"}
CONTROLLER_SHA256 = digest(Path(__file__).resolve())

# Runs in an isolated interpreter, importing ONLY the frozen execution source.
# Even failed accounting retains the raw response and six reconstructed outputs
# where reconstruction is possible; unknown costs are never filled with zero.
COLLECT = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from memory_tuner import collect_offload_factorial as c
from memory_tuner.device_contract import write_new_json
source, stage = map(Path, sys.argv[1:3])
raw, records, errors = stage / "raw", None, []
try:
    records = c.accounting.capture((stage / "sacct.txt").read_text(), c.block_claims(raw))
    write_new_json(stage / "accounting.json", records)
except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
    errors.append("accounting capture: " + str(exc))
try:
    result = c.collect(raw, source, sys.argv[3], sys.argv[4], records)
except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
    errors.append("accounted reconstruction: " + str(exc))
    result = c.collect(raw, source, sys.argv[3], sys.argv[4])
c.write_results(result, stage / "results")
if errors:
    write_new_json(stage / "accounting-errors.json", {"errors": errors})
    raise SystemExit(2)
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    env.update(TZ="UTC", SLURM_TIME_FORMAT="standard", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               REVISION_SOURCE_ROOT=str(SOURCE), SOURCE_COMMIT=COMMIT,
               OFFLOAD_FACTORIAL_FREEZE_SHA256=FREEZE)
    return env


def command(args):
    return subprocess.run(args, cwd=PROJECT, env=environment(), check=True,
                          text=True, capture_output=True, timeout=30).stdout


def provenance():
    return dict(source=str(SOURCE), source_commit=COMMIT, freeze_sha256=FREEZE,
                controller_sha256=CONTROLLER_SHA256,
                first_job_id=FIRST, pid=os.getpid(), timestamp_ns=time.time_ns())


def save(path, **fields):
    write_new_json(path, dict(provenance(), **fields))


def read(path):
    return json.loads(path.read_text())


def bound_record(path):
    value = read(path)
    require(all(value.get(k) == v for k, v in provenance().items()
                if k not in {"pid", "timestamp_ns"}), f"Wrong workflow identity: {path}")
    return value


def log(event, **fields):
    record = dict(provenance(), event=event, **fields)
    with (STATE / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    sync_directory(STATE)
    print(json.dumps(record, sort_keys=True), flush=True)


def verify_source():
    require(command(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]).strip() == COMMIT,
            "Frozen source HEAD differs")
    require(not command(["git", "-C", str(SOURCE), "status", "--porcelain",
                         "--untracked-files=all"]).strip(), "Frozen source is dirty")
    require(digest(SOURCE / "benchmark/offload_factorial/freeze.json") == FREEZE,
            "Freeze anchor differs")


def targets(receipt=None):
    expected = {f"{FIRST}_0": 0}
    if receipt is not None:
        job = receipt["array_job_id"]
        require(isinstance(job, str) and re.fullmatch(r"[1-9]\d*", job)
                and job != FIRST, "Invalid release receipt job ID")
        expected.update({f"{job}_{i}": i for i in range(1, 18)})
    return expected


def query(expected):
    jobs = ",".join(dict.fromkeys(job.split("_")[0] for job in expected))
    widths = {"JobID": 64, "JobIDRaw": 64, "JobName": 128, "State": 64,
              "AllocTRES": 512, "NodeList": 256, "Reason": 512}
    fields = ",".join(f"{key}%{widths[key]}" if key in widths else key for key in FIELDS)
    return command(["sacct", "--local", "--array", "-n", "-P", "-j", jobs,
                    "--starttime=2026-09-17T00:00:00", "--format=" + fields])


def terminal_status(text, expected):
    outer, seen, all_terminal = {}, set(), True
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        require(len(values) == len(FIELDS), "Malformed scheduler response; retained workflow stops")
        row = dict(zip(FIELDS, values))
        job, _, step = row["JobID"].partition(".")
        require(job in expected and step in {"", "batch", "extern"},
                f"Unexpected scheduler identity: {row['JobID']}")
        require(row["JobIDRaw"] not in seen, "Duplicate scheduler record")
        seen.add(row["JobIDRaw"])
        state = row["State"].split()[0] if row["State"].split() else ""
        all_terminal &= state in TERMINAL_STATES
        if not step:
            require(job not in outer and re.fullmatch(r"[1-9]\d*", row["JobIDRaw"]),
                    "Duplicate/invalid outer allocation")
            outer[job] = state
    return set(outer) == set(expected) and all_terminal, dict(Counter(outer.values()))


def inventory(root, *, sync=False):
    require(root.is_dir() and not root.is_symlink(), f"Missing/linked raw directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()),
                f"Linked/special evidence: {path}")
        if path.name == "snapshot-receipt.json" and path.parent == root:
            continue
        result[str(path.relative_to(root))] = digest(path) if path.is_file() else None
        if sync and path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    if sync:
        for path in reversed([root, *(p for p in root.rglob("*") if p.is_dir())]):
            sync_directory(path)
    return result


def collect_child(stage):
    args = [str(PROJECT / ".venv-reproduce/bin/python"), "-I", "-B", "-c", COLLECT,
            str(SOURCE), str(stage), FREEZE, COMMIT]
    with (stage / "collector.log").open("x") as output:
        with subprocess.Popen(args, cwd=SOURCE, env=environment(),
                              stdout=output, stderr=subprocess.STDOUT) as child:
            try:
                while child.poll() is None:
                    time.sleep(1)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
                raise
            output.flush()
            os.fsync(output.fileno())
            require(child.returncode == 0,
                    f"Reconstruction/accounting failed; inspect {stage}/collector.log and retained evidence")


def reconstruct(name, text):
    stage = STATE / name
    receipt = stage / "snapshot-receipt.json"
    if receipt.exists():
        require(bound_record(receipt)["files"] == inventory(stage), "Changed retained reconstruction")
        return stage
    require(not stage.exists(), f"Incomplete reconstruction retained at {stage}; human review required")
    stage.mkdir()
    with (stage / "sacct.txt").open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    original, copied = PROJECT / RAW, stage / "raw" / RAW
    require(original.resolve() == original, "Linked live raw root")
    before = inventory(original) if original.exists() else None
    if before is None:
        copied.mkdir(parents=True)
    else:
        shutil.copytree(original, copied, symlinks=True)
        require(before == inventory(original) == inventory(copied), "Raw changed during relocation")
    write_new_json(stage / "raw-inventory.json", before)
    collect_child(stage)
    require({p.name for p in (stage / "results").iterdir()} == RESULTS,
            "Collector must retain exactly six results")
    save(receipt, files=inventory(stage, sync=True))
    return stage


def check_results(stage, expected, *, gate):
    summary = read(stage / "results/summary.json")
    with (stage / "results/slots.csv").open(newline="") as handle:
        slots = list(csv.DictReader(handle))
    require(len(slots) == 144 and len({r["experiment_id"] for r in slots}) == 144
            and Counter(int(r["block_index"]) for r in slots) == Counter({i: 8 for i in range(18)}),
            "Reconstruction does not preserve all 144 slots")
    require(summary["freeze_sha256"] == FREEZE and summary["execution_source_commit"] == COMMIT
            and summary["independent_raw_reconstruction"] is True
            and summary["source_and_predictions_verified"] is True
            and Path(summary["raw_root"]) == stage / "raw", "Wrong reconstruction provenance")
    accounting = read(stage / "accounting.json")
    require({a["scheduler_job_id"] for a in accounting} == set(expected)
            and len(accounting) == len(expected), "Accounting is not the exact submitted allocation set")
    for item in accounting:
        index = expected[item["scheduler_job_id"]]
        require({r["block_id"] for r in slots if int(r["block_index"]) == index} == {item["pair_id"]},
                "Accounting array index differs from claimed block")
    costs = summary["allocation_accounting"]
    require(costs.get("allocations") == len(expected)
            and math.isfinite(costs["whole_allocation_gpu_seconds"])
            and costs["whole_allocation_gpu_seconds"] >= 0, "Missing actual whole-allocation cost")
    if gate:
        require(summary["claimed_blocks"] == 1 and not summary["block_validation_errors"],
                "First-block wrapper invalid or unexpected additional claims")
        first = [r for r in slots if int(r["block_index"]) == 0]
        require({int(r["period"]) for r in first} == set(range(1, 9)), "Missing first-block period")
        invalid = [r["experiment_id"] for r in first if r["eligible"] != "True"
                   or r["status"] not in {"completed", "memory_failure"}
                   or json.loads(r["validation_errors"]) or json.loads(r["cost_validation_errors"])
                   or not r["task_gpu_seconds"] or not math.isfinite(float(r["task_gpu_seconds"]))
                   or float(r["task_gpu_seconds"]) <= 0]
        require(not invalid, f"First-block execution gate failed: {invalid}; inspect {stage}/results/slots.csv")
        block = stage / "raw" / RAW / "blocks" / first[0]["block_id"]
        ledger = [json.loads(line) for line in (block / "ledger.jsonl").read_text().splitlines()]
        terminal = {k: v for k, v in ledger[-1].items()
                    if k not in {"event", "sequence", "timestamp_ns", "monotonic_ns"}}
        require(ledger[-1]["event"] == "block_terminal" and terminal == read(block / "block-summary.json")
                and terminal["planned_invocations"] == terminal["launched_invocations"] == 8,
                "First block lacks complete durable wrapper termination")
    return summary


def release(stage):
    require(not (STATE / "submit-intent.json").exists(), "Existing submission intent: NEVER retry")
    verify_source()
    require(read(stage / "raw-inventory.json") == inventory(PROJECT / RAW),
            "Live evidence/claims changed after the first-block gate")
    args = ["sbatch", "--parsable", "--array=1-17%3", "--account=OPEN-35-44",
            "--partition=qgpu", "--nodes=1", "--ntasks=1", "--gpus=4",
            "--cpus-per-task=64", "--mem=384G", "--time=03:00:00", "--no-requeue",
            "--export=ALL", str(SOURCE / "memory_tuner/run_offload_factorial.slurm")]
    save(STATE / "submit-intent.json", command=args,
         gate_receipt_sha256=digest(stage / "snapshot-receipt.json"))
    # Any exception/timeout/invalid stdout after intent is ambiguous. The durable
    # intent forbids ALL future submit attempts, including after process death.
    try:
        reply = command(args)
    except (subprocess.SubprocessError, OSError) as exc:
        save(STATE / "submit-response.json", error=str(exc),
             stdout=str(getattr(exc, "stdout", None)), stderr=str(getattr(exc, "stderr", None)))
        raise ValueError("Ambiguous submission; reconcile with scheduler manually, NEVER retry") from exc
    save(STATE / "submit-response.json", stdout=reply)
    require(re.fullmatch(r"[1-9]\d*\n?", reply) and reply.strip() != FIRST,
            "Ambiguous sbatch reply; reconcile with scheduler manually, NEVER retry")
    save(STATE / "submit-receipt.json", array_job_id=reply.strip(), indices="1-17%3")
    log("remaining_submitted", array_job_id=reply.strip())


def workflow(poll):
    last_status, last_log = None, 0.
    while True:
        require(not (STATE / "STOP.json").exists(), f"Stopped workflow: inspect {STATE}/STOP.json")
        if (STATE / "complete.json").exists():
            return bound_record(STATE / "complete.json")
        receipt = bound_record(STATE / "submit-receipt.json") if (STATE / "submit-receipt.json").exists() else None
        require(receipt is not None or not (STATE / "submit-intent.json").exists(),
                "Submission intent without receipt: reconcile with scheduler manually, NEVER retry")
        require(receipt is None or ((STATE / "submit-intent.json").exists()
                                   and receipt["indices"] == "1-17%3"), "Unbound submission receipt")
        expected = targets(receipt)
        try:
            text = query(expected)
        except (subprocess.SubprocessError, OSError) as exc:
            status, ready = {"query_error": str(exc)}, False
        else:
            ready, status = terminal_status(text, expected)
        if status != last_status or time.monotonic() - last_log >= 600:
            log("scheduler_status", status=status, expected_allocations=len(expected))
            last_status, last_log = status, time.monotonic()
        if not ready:
            time.sleep(poll)
            continue
        verify_source()
        stage = reconstruct("final" if receipt else "first", text)
        summary = check_results(stage, expected, gate=receipt is None)
        if receipt is None:
            log("first_block_gate_passed", results=str(stage / "results"))
            release(stage)
        else:
            save(STATE / "complete.json", results=str(stage / "results"),
                 accounting=str(stage / "accounting.json"), statuses=summary["statuses"])
            log("all_terminal_reconstructed", results=str(stage / "results"))
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--once", action="store_true")
    parser.add_argument("--background", action="store_true",
                        help="Detach the opt-in workflow; retain its PID and logs in the project")
    parser.add_argument("--poll-seconds", type=int, default=30, choices=range(1, 61))
    args = parser.parse_args(argv)
    require(STATE.is_relative_to(PROJECT) and STATE.resolve() == STATE,
            "State must remain in its fixed, non-linked project-local directory")
    require(not args.background or args.execute, "--background requires --execute")
    if args.background:
        STATE.mkdir(parents=True, exist_ok=True)
        save(STATE / "background-intent.json", action="start fixed experiment workflow once")
        with (STATE / "background.log").open("x") as output:
            child = subprocess.Popen(
                [sys.executable, "-B", "-m", "memory_tuner.watch_offload_factorial",
                 "--execute", "--poll-seconds", str(args.poll_seconds)],
                cwd=Path(__file__).resolve().parents[1], env=environment(),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True)
        save(STATE / "background-receipt.json", child_pid=child.pid)
        print(json.dumps(dict(provenance(), child_pid=child.pid, state=str(STATE))))
        return 0
    if not args.execute:
        receipt = bound_record(STATE / "submit-receipt.json") if (STATE / "submit-receipt.json").exists() else None
        ready, status = terminal_status(query(targets(receipt)), targets(receipt))
        print(json.dumps(dict(provenance(), terminal=ready, status=status,
                              stopped=(STATE / "STOP.json").exists(),
                              intent=(STATE / "submit-intent.json").exists())))
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sync_directory(STATE)
        log("watcher_started")
        try:
            workflow(args.poll_seconds)
        except (Exception, KeyboardInterrupt) as exc:
            if not (STATE / "STOP.json").exists():
                save(STATE / "STOP.json", error=f"{type(exc).__name__}: {exc}",
                     action="Review retained evidence/intent; do not retry, cancel or replace jobs automatically")
            log("stopped", error=str(exc))
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
