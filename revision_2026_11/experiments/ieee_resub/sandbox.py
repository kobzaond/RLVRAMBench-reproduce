"""Local, side-effect-free execution of run_grpo_ieee_resub.slurm for dry runs/tests.

The real launcher runs under bash with stub `python` (port selection, snapshot
resolution, environment capture, GPU monitor, device-contract check) and a stub
`apptainer` that records its argv instead of starting a container. The only
change to the launcher text is that its two FTA-26-70 path assertions are
neutralized so that the roots can be temporary directories. Nothing is
submitted and no cluster path is touched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ieee_resub import common as C
from ieee_resub.env import checkpoint_dir, launcher_environment, run_name

PATH_ASSERTIONS = (
    '[[ "$RLVRAM_PROJECT_ROOT" == /mnt/proj1/fta-26-70/* ]]\n',
    '[[ "$RLVRAM_SCRATCH_ROOT" == /scratch/project/fta-26-70/* ]]\n',
)

STUB_PYTHON = r'''
import json, os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[:1] == ["-m"]:
    module, rest = args[1], args[2:]
else:
    module, rest = Path(args[0]).name, args[1:]
def value(flag):
    return rest[rest.index(flag) + 1]
if module == "select_port_block.py":
    print(61000)
elif module == "resolve_hf_snapshot.py":
    print(value("--model"))
elif module == "capture_environment.py":
    Path(value("--output")).write_text(json.dumps({"sandbox": True}))
elif module == "device_contract.py":
    uuids = os.environ["RLVRAM_GPU_UUIDS"].split(",")
    assert len(uuids) == int(os.environ["N_GPUS_PER_NODE"]), "task GPU count"
elif module == "gpu_monitor.py":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    uuids = value("--gpu-uuids").split(",")
    Path(value("--device-map")).write_text(json.dumps({"uuids": uuids}))
    with open(value("--memory-csv"), "w") as handle:
        handle.write("%d,%s\n" % (time.time_ns(), ",".join(["1000"] * len(uuids))))
    Path(value("--phase-csv")).write_text("")
    Path(value("--telemetry-csv")).write_text("")
    while True:
        time.sleep(0.05)
else:
    raise SystemExit("unexpected python call: %r" % (args,))
'''

STUB_APPTAINER = r'''#!/usr/bin/env bash
set -euo pipefail
"$SANDBOX_REAL_PYTHON" -I -B "$SANDBOX_STUB_APPTAINER_PY" "$SANDBOX_CAPTURE/apptainer-$(date +%s%N).json" "$@"
echo "sandbox apptainer: no container started"
'''

# Records argv and the private-/tmp marker that the container would see at /tmp
# (the content of .rlvram-private-tmp in the source bound onto /tmp).
STUB_APPTAINER_PY = r'''
import json, sys
from pathlib import Path
argv = sys.argv[2:]
marker = None
for i, option in enumerate(argv[:-1]):
    if option == "--bind" and argv[i + 1].endswith(":/tmp"):
        path = Path(argv[i + 1].rsplit(":", 1)[0]) / ".rlvram-private-tmp"
        marker = path.read_text().strip() if path.is_file() else None
json.dump(dict(argv=argv, private_tmp_marker=marker), open(sys.argv[1], "w"))
'''


def prepare(root: Path) -> dict:
    """Create the temporary roots, stubs and input placeholders."""
    paths = dict(project=root / "proj", scratch=root / "scratch", source=root / "src",
                 output=root / "proj/output", bin=root / "bin", capture=root / "capture")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (paths["project"] / "images").mkdir(exist_ok=True)
    (paths["project"] / "images" / C.CONTAINER_NAME).write_text("sandbox image placeholder\n")
    queue = paths["source"] / "third_party/TransferQueue/transfer_queue"
    queue.mkdir(parents=True, exist_ok=True)
    (queue / "__init__.py").write_text("# sandbox\n")
    for dataset, subdir in C.DATA_SUBDIRS.items():
        data = root / "data" / subdir
        data.mkdir(parents=True, exist_ok=True)
        for split in ("train.parquet", "test.parquet"):
            (data / split).write_text("sandbox\n")
    (paths["bin"] / "stub_python.py").write_text(STUB_PYTHON)
    python = paths["bin"] / "python"
    python.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" -I -B "{paths["bin"] / "stub_python.py"}" "$@"\n')
    python.chmod(0o755)
    (paths["bin"] / "stub_apptainer.py").write_text(STUB_APPTAINER_PY)
    apptainer = paths["bin"] / "apptainer"
    apptainer.write_text(STUB_APPTAINER)
    apptainer.chmod(0o755)
    launcher = (C.REPO_ROOT / C.LAUNCHER).read_text()
    for line in PATH_ASSERTIONS:
        C.require(launcher.count(line) == 1, "Launcher path assertion changed")
        launcher = launcher.replace(line, "true  # sandbox: path assertion neutralized\n")
    (paths["bin"] / "launcher.sh").write_text(launcher)
    return paths


def render(row: dict, paths: dict, *, job: str = "9000001", period: int | None = None,
           create_checkpoint: bool = True, preserved_source: str | None = None) -> dict:
    """Run the launcher for one matrix row; return env, apptainer argv and trial record.

    preserved_source: relative path below <scratch>/preserved-checkpoints/ of a
    restore source (amendment reruns) instead of the block's own long run."""
    root = paths["project"].parent
    gpus = int(row["gpu_count"])
    allocation = [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in range(int(row["allocation_gpus"]))]
    task = [allocation[int(i)] for i in row["gpu_subset_ordinals"].split(";")]
    snapshot = root / "snapshot"
    snapshot.mkdir(exist_ok=True)
    resume = None
    if row["resume_mode"] == "resume_path":
        source = dict(row, experiment_id=row["resume_source_experiment_id"])
        resume = checkpoint_dir(source, job, paths["scratch"]) / f"global_step_{int(row['resume_step'])}"
        if preserved_source is not None:
            resume = paths["scratch"] / "preserved-checkpoints" / preserved_source
        if create_checkpoint:
            (resume / "actor").mkdir(parents=True, exist_ok=True)
            (resume / "data.pt").write_text("sandbox\n")
    attempt = paths["output"] / run_name(row, job)
    values = launcher_environment(
        row, job_id=job, source_root=paths["source"], output_root=paths["output"], attempt=attempt,
        invocation=f"{row['experiment_id']}-j{job}", period=period or int(row["period"]), task_uuids=task,
        allocation_uuids=allocation, snapshot=snapshot, resume_from=resume,
        project_root=paths["project"], scratch_root=paths["scratch"])
    values["RLVRAM_INSTRUMENTED_PACKAGES"] = str(C.REPO_ROOT / C.INSTRUMENTED_PACKAGES)
    values["DATA_DIR"] = str(root / "data" / C.DATA_SUBDIRS[row["dataset"]])
    env = {key: os.environ[key] for key in ("HOME", "USER", "LANG") if key in os.environ}
    env.update(values, PATH=f"{paths['bin']}:/usr/bin:/bin", SLURM_JOB_ID=job,
               SANDBOX_CAPTURE=str(paths["capture"]), SANDBOX_REAL_PYTHON=sys.executable,
               SANDBOX_STUB_APPTAINER_PY=str(paths["bin"] / "stub_apptainer.py"))
    env.setdefault("USER", "kobzaond")
    before = set(paths["capture"].glob("apptainer-*.json"))
    # Files, not pipes: the launcher's watchdog can leave a short-lived `sleep`
    # holding inherited descriptors, which would delay EOF on a pipe.
    log = root / f"launcher-{row['experiment_id']}.log"
    with log.open("w") as handle:
        completed = subprocess.run(["bash", str(paths["bin"] / "launcher.sh")], env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True, timeout=120)
    completed.stderr = log.read_text()
    new = sorted(set(paths["capture"].glob("apptainer-*.json")) - before)
    trial_path = attempt / f"trial-{job}.json"
    result = dict(experiment_id=row["experiment_id"], returncode=completed.returncode,
                  stderr_tail=completed.stderr[-2000:], environment=values)
    if new:
        record = json.loads(new[-1].read_text())
        argv = record["argv"]
        image = str(paths["project"] / "images" / C.CONTAINER_NAME)
        cut = argv.index(image)
        result["apptainer_options"] = argv[:cut]
        result["trainer_command"] = argv[cut + 1:]
        result["private_tmp_marker"] = record["private_tmp_marker"]
    if trial_path.is_file():
        result["trial"] = json.loads(trial_path.read_text())
    return result


def new_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="ieee-resub-sandbox-"))
