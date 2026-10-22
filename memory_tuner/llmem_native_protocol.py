"""Freeze and execute supported-architecture LLMem validation pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from memory_tuner.device_contract import write_new_json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/"benchmark/electronics_revision_20261016/llmem_native"
MODELS = {
    "opt-125m": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    "opt-1.3b": "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62",
}
PYTHON = "/home/kobzaond/rlvrambench-llmem-env/bin/python"
OUTPUT = Path("/mnt/proj3/open-35-44/verl/rl/output/llmem_native_validation_20261016")


def digest(path):
    # The upstream environment intentionally runs Python 3.10.
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def package_hashes():
    import colossalai
    base = Path(colossalai.__file__).parent
    return {str(p.relative_to(base)): digest(p) for p in sorted(base.rglob("*.py"))}


def freeze():
    DIRECTORY.mkdir(exist_ok=False)
    rows = []
    models = {}
    for model, revision in MODELS.items():
        path = Path("/home/kobzaond/rlvrambench-external/models") / (
            "models--facebook--"+model) / "snapshots" / revision
        models[model] = dict(path=str(path), revision=revision,
                            sha256={name: digest(path/name) for name in ("config.json", "pytorch_model.bin")})
        for batch in (1, 4):
            for sequence in (256, 512):
                block_index = len(rows)//3
                for seed in (441, 442, 443):
                    rows.append(dict(experiment_id=f"{model}-b{batch}-l{sequence}-s{seed}",
                                     model=model, batch=batch, sequence=sequence, seed=seed,
                                     block_index=block_index))
    with (DIRECTORY/"matrix.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = dict(study="Unmodified LLMem estimator on supported OPT/Gemini FP16 training",
                    models=models, matrix_sha256=digest(DIRECTORY/"matrix.csv"),
                    upstream_estimator_sha256=digest(ROOT/"third_party/llmem_estimator/size_estimator.py"),
                    colossalai_files_sha256=package_hashes(),
                    package_versions=subprocess.check_output([PYTHON,"-m","pip","freeze"],text=True).splitlines(),
                    planned_pairs=24, blocks=8, max_concurrent_allocations=2,
                    wall_seconds=3300, subprocess_timeout_seconds=600,
                    measurement="10-ms per-device NVML; whole-process and training-only maxima; allocator peaks separate",
                    pairing="Predict with measured initialization first; measure three updates in a fresh process on the same GPU",
                    synthetic_input="Uniform nonpadding IDs sampled reproducibly for specified shapes; labels clone input",
                    analysis="Per-case and equal-case signed error, MAE, relative error, and underestimation; retain all unresolved pairs",
                    limitations="Native training only. No LoRA/FSDP/vLLM adaptation, RL-loop approval claim, or reproduction of the original accuracy table.",
                    retries=0, development_excluded=["5214860","5214872"],
                    frozen_at_epoch=time.time())
    write_new_json(DIRECTORY/"protocol.json", protocol)
    print(json.dumps(dict(protocol_sha256=digest(DIRECTORY/"protocol.json"),pairs=len(rows))))


def run(args):
    assert digest(DIRECTORY/"protocol.json") == args.protocol_sha256
    protocol=json.loads((DIRECTORY/"protocol.json").read_text())
    assert digest(DIRECTORY/"matrix.csv") == protocol["matrix_sha256"]
    assert package_hashes() == protocol["colossalai_files_sha256"]
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip() == args.source_commit
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    with (DIRECTORY/"matrix.csv").open() as handle:
        rows=[r for r in csv.DictReader(handle) if int(r["block_index"])==args.index]
    assert len(rows)==3 and os.environ["SLURM_ARRAY_TASK_ID"]==str(args.index)
    assert os.environ.get("SLURM_RESTART_COUNT","0")=="0"
    block = OUTPUT/("block-"+str(args.index))
    block.mkdir(parents=True, exist_ok=False)
    uuids=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).splitlines()
    assert len(uuids)==1 and uuids[0].startswith("GPU-")
    uuid=uuids[0]
    model=protocol["models"][rows[0]["model"]]
    for name, expected in model["sha256"].items():
        assert digest(Path(model["path"])/name)==expected
    write_new_json(block/"claim.json",dict(rows=rows, source_commit=args.source_commit,
                    protocol_sha256=args.protocol_sha256, job_id=os.environ["SLURM_JOB_ID"],gpu_uuid=uuid))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid)
    failed = False
    for row in rows:
        pair=block/row["experiment_id"]
        pair.mkdir(exist_ok=False)
        statuses={}
        for mode in ("estimate","measure"):
            # An unsupported estimate does not prevent independently observing
            # training, but neither part may overwrite an earlier attempt.
            command=[PYTHON,"-m","memory_tuner.llmem_native","--mode",mode,
                     "--model",model["path"],"--batch",row["batch"],"--sequence",row["sequence"],
                     "--seed",row["seed"],"--gpu-uuid",uuid,"--output",str(pair/mode),
                     "--port",str(20000+int(os.environ["SLURM_JOB_ID"])%20000)]
            with (pair/(mode+".log")).open("x") as log:
                started=time.time_ns()
                try:
                    finished=subprocess.run(command,env=environment,cwd=ROOT,stdout=log,
                                            stderr=subprocess.STDOUT,timeout=600)
                    statuses[mode]=dict(exit_code=finished.returncode,started_ns=started,finished_ns=time.time_ns())
                except subprocess.TimeoutExpired:
                    statuses[mode]=dict(exit_code=None,timeout=True,started_ns=started,finished_ns=time.time_ns())
                failed |= statuses[mode]["exit_code"] != 0
            time.sleep(2)
        write_new_json(pair/"terminal.json",dict(row=row, executions=statuses,
            files_sha256={str(p.relative_to(pair)):digest(p) for p in pair.rglob("*") if p.is_file()}))
    return int(failed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","run"))
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--source-commit")
    parser.add_argument("--index",type=int)
    args=parser.parse_args()
    if args.command=="freeze":
        freeze()
        return 0
    return run(args)


if __name__=="__main__":
    raise SystemExit(main())
