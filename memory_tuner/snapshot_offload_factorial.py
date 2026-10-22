"""Export the exact offload-factorial source and CPU replay inputs, without Git.

This publication helper is outside the sealed execution closure. It neither
changes the freeze nor copies new target outcomes, model/data payloads, or Git.
Only the specified 3d6ecac source / 5df43f35 freeze pair is supported.

OUTPUT contains source/ (unchanged sealed runtime and loader inputs, plus
notices), snapshot_offload_factorial.py (this standalone verifier), and
source-export-report.json. The usual destination is
reproduction_sources/offload_factorial. Keep the returned report_sha256 in the
parent publication manifest; --report-sha256 verifies that external binding.
The source commit is the externally specified identity associated with this
freeze, not a claim that every file in a Git checkout was exported.

source_plan(source, freeze_sha256, source_commit) is read-only.
export_sources(source, freeze_sha256, source_commit, output, python=...) creates
one new destination, validates the unmodified loader in a subprocess, and
writes a success receipt last. Failure may leave an incomplete new directory;
it is never overwritten, repaired, or presented as a completed export.
verify_export(export, freeze_sha256, source_commit, python=..., report_sha256=...)
works after relocation, using only this export and an existing scientific
Python environment. Neither the original project nor its Git, models, data, or
raw runs is needed. CPU validation replays the plan and predictions, not GPU
outcomes. It sets PYTHONPATH to source/ and OMP/OPENBLAS/MKL thread counts to one.
No environment is installed or created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


SCHEMA = "offload-source-export-1"
SOURCE_COMMIT = "3d6ecac1852e16a0422c9c72a230187be663f8b5"
FREEZE_SHA256 = "5df43f35e0b02c23e4816b7e860c5a613176da41f5adb27e57f79ac2becf1fb9"
FREEZE_DIRECTORY = "benchmark/offload_factorial"
FREEZE_FILES = ("freeze.json", "protocol.json", "matrix.csv", "predictions.json")
SOURCE_DIRECTORY = "source"
REPORT_NAME = "source-export-report.json"
HELPER_NAME = "snapshot_offload_factorial.py"
RUNTIME_COUNT = 584
INPUT_NAMES = frozenset((
    "benchmark.py", "benchmark/attempts.csv", "benchmark/configurations.csv", "benchmark/outcomes.csv",
    "benchmark/completion_replication/freeze.json", "benchmark/completion_replication/protocol.json",
    "benchmark/estimation/amendment-01.json", "benchmark/estimation/fitted_model.json",
    "benchmark/estimation/matrix.csv", "benchmark/estimation/model_metadata.json",
    "benchmark/estimation/prediction_freeze.json", "benchmark/estimation/predictions.json",
    "benchmark/estimation/protocol.json", "benchmark/estimation/retrospective.json",
    "benchmark/estimation/targets.csv", "memory_tuner/estimation_baselines.py",
    "memory_tuner/model_memory_metadata.py", "memory_tuner/plan_estimation_study.py",
))
REQUIRED_RUNTIME = frozenset((
    "benchmark.py", "run_grpo_instrumented.slurm", "memory_tuner/__init__.py",
    "memory_tuner/offload_factorial_protocol.py", "memory_tuner/offload_factorial_runtime.py",
    "memory_tuner/run_offload_factorial.py", "memory_tuner/run_offload_factorial.slurm",
    "memory_tuner/collect_offload_factorial.py", "memory_tuner/offload_factorial_analysis.py",
    "memory_tuner/completion_replication_protocol.py", "memory_tuner/estimation_baselines.py",
    "memory_tuner/device_contract.py", "memory_tuner/model_memory_metadata.py",
    "memory_tuner/resolve_hf_snapshot.py", "sppo_replay/allocator_trace.py",
    "third_party/TransferQueue/transfer_queue/__init__.py",
))
REQUIRED_NOTICES = (
    "LICENSE-RLVRAMBench", "instrumented_python_packages/verl/LICENSE", "third_party/TransferQueue/LICENSE",
)
NOTICE_ROOTS = ("instrumented_python_packages/verl", "third_party/TransferQueue")
PAYLOAD_SUFFIXES = frozenset((
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".h5",
    ".npy", ".npz", ".sif", ".simg", ".img", ".parquet", ".arrow",
    ".tar", ".zip", ".gz", ".zst", ".so", ".pyc",
))
MAX_TEXT_BYTES = 32 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def anchors(freeze_sha256, source_commit):
    require(freeze_sha256 == FREEZE_SHA256, "Wrong external offload freeze anchor")
    require(source_commit == SOURCE_COMMIT, "Wrong externally specified offload source commit")


def explicit_path(value, *, directory=False):
    require(value is not None, "Explicit path required; no recorded/live fallback")
    path = Path(value)
    require(".." not in path.parts, f"Escaping path: {path}")
    path = path.absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"Linked path: {path}")
    if directory:
        require(path.is_dir(), f"Missing directory: {path}")
    return path


def relative_name(name):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name)
            and Path(name).as_posix() == name and not {".", ".."} & set(Path(name).parts),
            f"Unsafe relative source name: {name!r}")
    require(not {".git", ".venv", "__pycache__"} & set(Path(name).parts)
            and Path(name).suffix.lower() not in PAYLOAD_SUFFIXES,
            f"Git/cache/model/data/container payload forbidden: {name}")
    return name


def read_bytes(root, name, expected=None):
    """Read a bounded, unlinked regular text file; reject file changes and links."""
    relative_name(name)
    root = explicit_path(root, directory=True)
    path = explicit_path(root / name)
    require(path.is_relative_to(root), f"Entry escapes root: {name}")
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                f"Entry must be an unlinked regular file: {name}")
        require(before.st_size <= MAX_TEXT_BYTES, f"Oversized source/input: {name}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            payload = handle.read(MAX_TEXT_BYTES + 1)
            after = os.fstat(handle.fileno())
        explicit_path(path)
        current = path.lstat()
        signature = lambda info: (info.st_dev, info.st_ino, info.st_size,
                                  info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)
        require(signature(before) == signature(after) == signature(current)
                and len(payload) == before.st_size, f"Source/input changed while reading: {name}")
    except OSError as error:
        raise ValueError(f"Cannot read required regular source/input {name}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    require(b"\0" not in payload, f"Binary payload is not source/metadata: {name}")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Non-text payload is not source/metadata: {name}") from error
    if expected is not None:
        require(valid_hash(expected) and sha256(payload) == expected, f"Source/input hash differs: {name}")
    return payload


def parse_json(payload):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(payload, object_pairs_hook=unique, parse_constant=nonfinite)


def merge_hashes(target, source):
    require(isinstance(source, dict), "Expected a relative-name/SHA256 inventory")
    for name, value in source.items():
        relative_name(name)
        require(valid_hash(value), f"Invalid SHA256 for {name}")
        require(name not in target or target[name] == value, f"Conflicting sealed input: {name}")
        target[name] = value


def loader_inputs(root, declared):
    """Follow exactly the inherited metadata reads made by the frozen loader.

    The effective prediction seal includes the historical feature amendment.
    Recursing into the earlier protocol's superseded implementation hash would
    incorrectly demand obsolete code. Earlier cohort runtime seals and model/
    parquet inventories are metadata, not the offload CPU loader's read set.
    """
    prefix = "benchmark/estimation/"
    seal_name = prefix + "prediction_freeze.json"
    seal = parse_json(read_bytes(root, seal_name, declared[seal_name]))
    result = {}
    merge_hashes(result, seal["effective_input_sha256"])
    for field in ("files_sha256", "amendments_sha256"):
        merge_hashes(result, {prefix + name: value for name, value in seal[field].items()})
    merge_hashes(result, {
        seal_name: declared[seal_name], prefix + "protocol.json": seal["protocol_sha256"],
        "memory_tuner/estimation_baselines.py": seal["implementation_sha256"],
    })
    inherited_prefix = "benchmark/completion_replication/"
    inherited_name = inherited_prefix + "freeze.json"
    inherited = parse_json(read_bytes(root, inherited_name, declared[inherited_name]))
    merge_hashes(result, {
        inherited_name: declared[inherited_name],
        inherited_prefix + "protocol.json": inherited["files_sha256"]["protocol.json"],
    })
    require(result == declared, "Transitive loader inputs differ from the sealed offload input closure")
    return dict(sorted(result.items()))


def notices(root):
    names = set(REQUIRED_NOTICES)
    for directory in NOTICE_ROOTS:
        for path in explicit_path(root / directory, directory=True).iterdir():
            if re.fullmatch(r"(LICENSE|LICENCE|NOTICE|COPYING|AUTHORS)(?:[._-].*)?",
                            path.name, re.IGNORECASE):
                names.add(path.relative_to(root).as_posix())
    return {name: sha256(read_bytes(root, name)) for name in sorted(names)}


def source_plan(source, freeze_sha256, source_commit):
    anchors(freeze_sha256, source_commit)
    source = explicit_path(source, directory=True)
    seal_name = FREEZE_DIRECTORY + "/freeze.json"
    seal_payload = read_bytes(source, seal_name, freeze_sha256)
    seal = parse_json(seal_payload)
    require(seal["schema_version"] == "offload-factorial-1"
            and seal["status"] == "sealed_before_offload_factorial_execution"
            and set(seal["files_sha256"]) == set(FREEZE_FILES) - {"freeze.json"},
            "Malformed offload freeze")
    freeze_hashes = {seal_name: freeze_sha256}
    merge_hashes(freeze_hashes, {FREEZE_DIRECTORY + "/" + name: value
                                for name, value in seal["files_sha256"].items()})
    protocol_name = FREEZE_DIRECTORY + "/protocol.json"
    protocol = parse_json(read_bytes(source, protocol_name, freeze_hashes[protocol_name]))
    runtime, inputs = protocol["runtime_sha256"], protocol["input_sha256"]
    require(isinstance(runtime, dict) and len(runtime) == RUNTIME_COUNT
            and REQUIRED_RUNTIME <= runtime.keys(), "Expected the complete 584-file offload runtime")
    require(isinstance(inputs, dict) and set(inputs) == INPUT_NAMES,
            "Expected the exact 18 inherited offload inputs")
    hashes = {}
    merge_hashes(hashes, runtime)
    merge_hashes(hashes, inputs)
    merge_hashes(hashes, freeze_hashes)
    for name in runtime:
        require(name in ("benchmark.py", "run_grpo_instrumented.slurm") or
                (name.startswith(("memory_tuner/", "instrumented_python_packages/",
                                  "third_party/TransferQueue/", "sppo_replay/"))
                 and Path(name).suffix in (".py", ".slurm")),
                f"Not an exportable runtime source entry: {name}")
    transitive = loader_inputs(source, inputs)
    attribution = notices(source)
    merge_hashes(hashes, attribution)
    files = {}
    for name, expected in sorted(hashes.items()):
        payload = read_bytes(source, name, expected)
        files[name] = dict(sha256=expected, size_bytes=len(payload))
    return dict(
        schema_version=SCHEMA, freeze_sha256=freeze_sha256, source_commit=source_commit,
        source_directory=SOURCE_DIRECTORY, runtime_sha256=dict(sorted(runtime.items())),
        input_sha256=dict(sorted(inputs.items())), transitive_loader_input_sha256=transitive,
        freeze_files_sha256=dict(sorted(freeze_hashes.items())), notices_sha256=attribution,
        files=files, source_file_count=len(files),
        source_total_bytes=sum(item["size_bytes"] for item in files.values()),
    )


# Fixed code supplied by this helper, never executed from a receipt field.
VALIDATE_CODE = r'''
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import numpy
import scipy
import sklearn

source = Path(sys.argv[1]).resolve()
freeze, commit = sys.argv[2:4]
if os.environ.get("PYTHONPATH") != str(source):
    raise RuntimeError("PYTHONPATH must contain only the exported source")
if any(os.environ.get(name) != "1" for name in
       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
    raise RuntimeError("Scientific thread limits differ")

project_roots = {
    "memory_tuner": source, "benchmark": source, "sppo_replay": source,
    "verl": source / "instrumented_python_packages",
    "transfer_queue": source / "third_party/TransferQueue",
}
class ClosedSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        first = fullname.split(".")[0]
        if first not in project_roots:
            return None
        search = list(path) if path is not None else [str(project_roots[first])]
        if not all(Path(p).resolve().is_relative_to(source) for p in search):
            raise ImportError("Live source fallback forbidden: " + fullname)
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None:
            raise ImportError("Missing exported source; live fallback forbidden: " + fullname)
        locations = ([spec.origin] if spec.origin and spec.origin not in ("built-in", "frozen")
                     else list(spec.submodule_search_locations or ()))
        if not all(Path(p).resolve().is_relative_to(source) for p in locations):
            raise ImportError("Foreign source module: " + fullname)
        return spec

sys.meta_path.insert(0, ClosedSourceFinder())
for name, module in tuple(sys.modules.items()):
    if name.split(".")[0] in project_roots:
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(source):
            raise ImportError("Previously imported live source forbidden: " + name)

# Dependencies use the supplied scientific interpreter; project/data/model reads
# outside source/ are forbidden during the unchanged loader's execution.
python_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
source_reads = set()
def guard(event, args):
    if event == "open":
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise PermissionError("CPU replay is read-only")
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if resolved.is_relative_to(source):
                source_reads.add(resolved.relative_to(source).as_posix())
            elif not any(resolved.is_relative_to(root) for root in python_roots):
                raise PermissionError("Live/project/model/data read forbidden: " + str(resolved))
    elif event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                   "socket.connect", "socket.bind", "os.remove", "os.rename",
                   "os.mkdir", "os.rmdir", "os.link", "os.symlink", "os.truncate"}:
        raise PermissionError("CPU replay forbids process/network/filesystem mutations: " + event)
sys.addaudithook(guard)

from memory_tuner import offload_factorial_protocol as design
if Path(design.__file__).resolve() != source / "memory_tuner/offload_factorial_protocol.py":
    raise RuntimeError("Not executing the exported offload loader")
frozen = design.load(source, freeze, verify_assets=False)
counts = (len(frozen["rows"]), len({r["block_id"] for r in frozen["rows"]}),
          len(frozen["protocol"]["targets"]), len(frozen["predictions"]))
if counts != (144, 18, 48, 288):
    raise RuntimeError("Incomplete offload plan/prediction replay")
modules = {}
for name, module in sorted(sys.modules.items()):
    if name.split(".")[0] in project_roots:
        origin = getattr(module, "__file__", None)
        if origin:
            modules[name] = Path(origin).resolve().relative_to(source).as_posix()
print(json.dumps(dict(
    status="passed", loader="source/memory_tuner/offload_factorial_protocol.py",
    planned_slots=144, planned_blocks=18, target_configurations=48, prediction_rows=288,
    freeze_sha256=freeze, source_commit=commit, verify_assets=False,
    prediction_replay="exact", outcome_reconstruction_performed=False,
    python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
    scikit_learn=sklearn.__version__, source_modules=modules,
    source_files_read=len(source_reads),
), sort_keys=True))
'''


def validate_cpu(export, freeze_sha256, source_commit, python):
    source = explicit_path(Path(export) / SOURCE_DIRECTORY, directory=True)
    # Resolving this symlink can discard the caller's virtual-environment identity.
    python = Path(python).absolute()
    require(python.is_file() and os.access(python, os.X_OK), "Existing scientific Python executable required")
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
                                           "SYSTEMROOT", "LD_LIBRARY_PATH") if key in os.environ}
    env.update(PYTHONPATH=str(source), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [str(python), "-B", "-s", "-P", "-c", VALIDATE_CODE,
               str(source), freeze_sha256, source_commit]
    try:
        result = subprocess.run(command, cwd=source, env=env, text=True, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise ValueError("Exported CPU loader timed out; no success receipt was written") from error
    require(result.returncode == 0, "Exported CPU loader failed. Use an existing project scientific "
            "Python (NumPy/SciPy/scikit-learn) and the complete source snapshot.\n"
            + (result.stderr or result.stdout)[-6000:])
    checked = parse_json(result.stdout)
    require(checked["status"] == "passed" and checked["freeze_sha256"] == freeze_sha256
            and checked["source_commit"] == source_commit and checked["prediction_replay"] == "exact",
            "Unexpected exported CPU validation result")
    return checked


def helper_record():
    path = Path(__file__).absolute()
    payload = read_bytes(path.parent, path.name)
    return dict(path=HELPER_NAME, sha256=sha256(payload), size_bytes=len(payload),
                validation_program_sha256=sha256(VALIDATE_CODE.encode()))


def verify_tree(export, plan, code, *, report_present):
    expected = {SOURCE_DIRECTORY + "/" + name: item for name, item in plan["files"].items()}
    expected[HELPER_NAME] = code
    directories = {parent.as_posix() for name in expected for parent in Path(name).parents
                   if parent != Path(".")}
    files_seen, dirs_seen = set(), set()
    for current, children, files in os.walk(export, followlinks=False):
        current = Path(current)
        for child in children:
            path = explicit_path(current / child, directory=True)
            dirs_seen.add(path.relative_to(export).as_posix())
        for name in files:
            relative = (current / name).relative_to(export).as_posix()
            require(relative in expected or (report_present and relative == REPORT_NAME),
                    f"Unexpected snapshot file: {relative}")
            record = expected.get(relative)
            payload = read_bytes(export, relative, record["sha256"] if record else None)
            if record:
                require(len(payload) == record["size_bytes"], f"Copied size differs: {relative}")
            files_seen.add(relative)
    require(files_seen == set(expected) | ({REPORT_NAME} if report_present else set())
            and dirs_seen == directories, "Missing or extra snapshot files/directories")


def write_new(path, payload):
    path = explicit_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    explicit_path(path.parent, directory=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def export_sources(source, freeze_sha256, source_commit, output, python=sys.executable):
    source = explicit_path(source, directory=True)
    output = explicit_path(output)
    require(not output.exists(), "Preserving existing destination; choose a new snapshot directory")
    require(not output.is_relative_to(source), "Snapshot destination must be outside execution source")
    plan = source_plan(source, freeze_sha256, source_commit)
    code = helper_record()
    output.mkdir(parents=True, exist_ok=False)
    for name, item in plan["files"].items():
        write_new(output / SOURCE_DIRECTORY / name, read_bytes(source, name, item["sha256"]))
    helper = Path(__file__).absolute()
    write_new(output / HELPER_NAME, read_bytes(helper.parent, helper.name, code["sha256"]))
    verify_tree(output, plan, code, report_present=False)
    checked = validate_cpu(output, freeze_sha256, source_commit, python)
    require(source_plan(source, freeze_sha256, source_commit) == plan, "Inputs changed during source export")
    require(helper_record() == code, "Publication helper changed during source export")
    verify_tree(output, plan, code, report_present=False)
    report = dict(
        plan, status="verified_git_free_cpu_export", verification_code=code, cpu_validation=checked,
        excludes=["Git metadata", "model/cache/container payloads", "workload parquet data", "raw target outcomes"],
        notice_provenance="Notices and the standalone verifier are publication additions, not entries "
        "in the execution freeze. Bind this entire receipt in the parent publication manifest.",
        scope="One offload execution cohort. Earlier prediction-seal metadata is retained only as "
        "needed by the unchanged offload loader. No amendment or earlier raw cohort is required.",
    )
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    write_new(output / REPORT_NAME, payload)
    require(read_bytes(output, REPORT_NAME, sha256(payload)) == payload, "Written receipt bytes differ")
    verify_tree(output, plan, code, report_present=True)
    return dict(report, report_sha256=sha256(payload))


def verify_export(export, freeze_sha256, source_commit, python=sys.executable, *, report_sha256=None):
    anchors(freeze_sha256, source_commit)
    export = explicit_path(export, directory=True)
    payload = read_bytes(export, REPORT_NAME, report_sha256)
    report = parse_json(payload)
    require(report["schema_version"] == SCHEMA and report["status"] == "verified_git_free_cpu_export",
            "Missing/invalid completed source-export receipt")
    plan = source_plan(export / SOURCE_DIRECTORY, freeze_sha256, source_commit)
    require(all(report.get(key) == value for key, value in plan.items()), "Receipt differs from bytes/anchor")
    code = helper_record()
    require(report["verification_code"] == code, "Receipt verification code differs; use the exported verifier")
    verify_tree(export, plan, code, report_present=True)
    checked = validate_cpu(export, freeze_sha256, source_commit, python)
    verify_tree(export, plan, code, report_present=True)
    require(read_bytes(export, REPORT_NAME) == payload, "Receipt changed during verification")
    return dict(status="verified_git_free_cpu_export", freeze_sha256=freeze_sha256,
                source_commit=source_commit, source_file_count=plan["source_file_count"],
                report_sha256=sha256(payload), cpu_validation=checked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "export", "verify"):
        child = commands.add_parser(name)
        child.add_argument("--freeze-sha256", required=True, help="Externally retained offload freeze SHA256")
        child.add_argument("--source-commit", required=True, help="Externally specified execution commit")
        if name == "verify":
            child.add_argument("--export-root", type=Path, required=True)
            child.add_argument("--report-sha256", help="Receipt SHA256 retained in the parent publication manifest")
        else:
            child.add_argument("--source-root", type=Path, required=True)
        if name != "preview":
            child.add_argument("--python", type=Path, default=Path(sys.executable),
                               help="Existing scientific Python; no environment is created")
        if name == "export":
            child.add_argument("--output", type=Path, required=True,
                               help="New directory, normally reproduction_sources/offload_factorial")
    args = parser.parse_args(argv)
    if args.command == "preview":
        result = source_plan(args.source_root, args.freeze_sha256, args.source_commit)
        result.update(status="source_bytes_verified_cpu_loader_not_run", writes_performed=False)
    elif args.command == "export":
        result = export_sources(args.source_root, args.freeze_sha256, args.source_commit, args.output, args.python)
    else:
        result = verify_export(args.export_root, args.freeze_sha256, args.source_commit,
                               args.python, report_sha256=args.report_sha256)
    inventories = {"files", "runtime_sha256", "input_sha256", "transitive_loader_input_sha256",
                   "freeze_files_sha256", "notices_sha256"}
    print(json.dumps({key: value for key, value in result.items() if key not in inventories},
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
