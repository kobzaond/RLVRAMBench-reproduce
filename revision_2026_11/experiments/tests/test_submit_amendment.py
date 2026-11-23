"""submit_ieee_resub.sh with an amendment, in a sandbox (stub hostname/git/sbatch; no cluster)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ieee_resub import common as C

SCRIPT = C.EXPERIMENT_DIR / "submit/submit_ieee_resub.sh"
RECORDED = C.EXPERIMENT_DIR / "amendments/e3-amendment-01.json"
COMMIT = "c" * 40
GROUP = C.GROUPS["e3"]


@pytest.fixture
def box(tmp_path):
    project, scratch, bin_dir = tmp_path / "proj", tmp_path / "scratch", tmp_path / "bin"
    for path in (project / "protocols/e3", project / "protocols/smoke2", project / "src", project / "venv/bin",
                 project / f"output/{GROUP}/blocks", bin_dir):
        path.mkdir(parents=True)
    (project / "src" / f"ieee-resub-{COMMIT[:12]}").symlink_to(C.REPO_ROOT)  # the tree under test
    (project / "protocols/smoke2/smoke2_gate.json").write_text("{}")
    stubs = {"hostname": "echo login1.karolina.it4i.cz",
             "git": f'case "$*" in *rev-parse*) echo {COMMIT} ;; *status*) ;; *) exit 9 ;; esac',
             "sbatch": f'printf "%s\\n" "$@" > {tmp_path}/sbatch-args; echo 7777777'}
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    (project / "venv/bin/python").symlink_to(sys.executable)
    amendment = json.loads(RECORDED.read_text().replace("/scratch/project/fta-26-70/rlvram", str(scratch))
                           .replace("/mnt/proj1/fta-26-70/rlvram", str(project)))
    before = dict(amendment["source_files_sha256"])
    for name, change in amendment["changed_source_files"].items():
        before.pop(name) if change["before"] is None else before.__setitem__(name, change["before"])
    protocol = dict(study="e3", status="frozen", group=GROUP, account="FTA-26-70", partition="qgpu",
                    caps=C.CAPS["e3"], source_files_sha256=before)
    protocol_path = project / "protocols/e3/e3_protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    amendment["protocol_sha256"] = C.digest(protocol_path)
    target = project / "protocols/e3/amendment-01.json"
    target.write_text(json.dumps(amendment))
    target.chmod(0o444)
    script = SCRIPT.read_text().replace("/mnt/proj1/fta-26-70/rlvram", str(project))
    (bin_dir / "submit.sh").write_text(script)
    env = dict(os.environ, PATH=f"{bin_dir}:/usr/bin:/bin")

    def submit(*extra, amendment_sha=None):
        args = ["bash", str(bin_dir / "submit.sh"), "e3", str(protocol_path), C.digest(protocol_path), COMMIT]
        if amendment_sha is not False:
            args += ["--amendment", str(target), amendment_sha or C.digest(target)]
        return subprocess.run(args + list(extra), env=env, capture_output=True, text=True, timeout=300)
    return dict(submit=submit, project=project, scratch=scratch, amendment=amendment, tmp=tmp_path)


def test_subset_of_never_launched_blocks(box):
    result = box["submit"]("--indices", "5-11")
    assert result.returncode == 0, result.stderr
    assert "--array=5-11%3" in result.stdout and "slurm/ieee_resub_e3.slurm" in result.stdout
    assert "AMENDMENT_SHA256=" in result.stdout and "print-only mode" in result.stdout
    assert box["submit"]("--indices", "4-11").returncode != 0  # 4 was launched under the original snapshot
    (box["project"] / f"output/{GROUP}/blocks/e3-q7-gsm8k-off1-r70-s632").mkdir()
    assert box["submit"]("--indices", "5-11").returncode != 0  # index 10 already claimed
    assert box["submit"]("--indices", "5-11", amendment_sha="0" * 64).returncode != 0
    assert box["submit"]("--indices", "5-11", amendment_sha=False).returncode != 0  # needs the amendment


def test_rerun_waits_for_a_terminal_unresolved_original_and_records_a_receipt(box):
    entry = box["amendment"]["reruns"][0]
    assert box["submit"]("--rerun", "--indices", "0").returncode != 0  # no terminal receipt yet
    block = box["project"] / f"output/{GROUP}/blocks" / entry["block_id"]
    block.mkdir()
    receipt = block / "period-3.json"
    receipt.write_text(json.dumps(dict(status="unresolved")))
    (block / "summary.json").write_text("{}")
    assert box["submit"]("--rerun", "--indices", "0").returncode != 0  # preserved checkpoint absent
    preserved = Path(entry["restore_source"]["folder"])
    preserved.mkdir(parents=True)
    (preserved / "data.pt").write_text("x")
    result = box["submit"]("--rerun", "--indices", "0")
    assert result.returncode == 0, result.stderr
    assert "--array=0%3" in result.stdout and "slurm/ieee_resub_e3_amendment.slurm" in result.stdout
    assert box["submit"]("--rerun").returncode != 0  # rerun 1 (s632-long) has not ended yet
    receipt.write_text(json.dumps(dict(status="within_margin")))
    assert box["submit"]("--rerun", "--indices", "0").returncode != 0  # completed outcomes are never rerun
    receipt.write_text(json.dumps(dict(status="unresolved")))
    submitted = box["submit"]("--rerun", "--indices", "0", "--submit")
    assert submitted.returncode == 0, submitted.stderr
    record = json.loads((box["project"] / "protocols/e3/submission-7777777.json").read_text())
    assert record["kind"] == "amendment-rerun" and record["array"] == "0%3"
    assert record["amendment_sha256"] == C.digest(box["project"] / "protocols/e3/amendment-01.json")
    args = (box["tmp"] / "sbatch-args").read_text().split("\n")
    assert args[:2] == ["--parsable", "--array=0%3"] and args[-2].endswith("ieee_resub_e3_amendment.slurm")
    (box["project"] / entry["claim"].replace("blocks/", f"output/{GROUP}/blocks/")).mkdir()
    assert box["submit"]("--rerun", "--indices", "0").returncode != 0  # the rerun claim is used once
    long_entry = box["amendment"]["reruns"][1]
    assert long_entry["role"] == "long" and "restore_source" not in long_entry
    long_block = box["project"] / f"output/{GROUP}/blocks" / long_entry["block_id"]
    long_block.mkdir()
    (long_block / "period-2.json").write_text(json.dumps(dict(status="unresolved")))
    assert box["submit"]("--rerun", "--indices", "1").returncode != 0  # block not finished
    (long_block / "summary.json").write_text("{}")
    result = box["submit"]("--rerun", "--indices", "1")
    assert result.returncode == 0, result.stderr
    assert "--array=1%3" in result.stdout
