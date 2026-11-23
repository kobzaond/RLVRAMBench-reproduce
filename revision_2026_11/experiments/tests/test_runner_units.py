"""Runner pieces that do not need an allocation: checkpoint release and restore preparation."""
import json

import pytest

from ieee_resub import common as C
from ieee_resub import plan, runner
from ieee_resub.env import checkpoint_dir


def fake_checkpoint(folder, gpus=4):
    (folder / "actor").mkdir(parents=True)
    (folder / "data.pt").write_text("x")
    for rank in range(gpus):
        for kind in ("model", "optim", "extra_state"):
            (folder / "actor" / f"{kind}_world_size_{gpus}_rank_{rank}.pt").write_text(kind)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CLUSTER_CHECKPOINT_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr(runner, "checkpoint_dir", lambda row, job: checkpoint_dir(row, job, tmp_path))
    return tmp_path


def block_rows():
    rows, _ = plan.e3_rows()
    block = next(r["block_id"] for r in rows if r["role"] == "restore")
    return [r for r in rows if r["block_id"] == block]


def test_restore_source_is_validated_frozen_and_hashed(scratch):
    rows = block_rows()
    long, restore = rows[1], rows[2]
    folder = checkpoint_dir(long, "77", scratch) / "global_step_50"
    results = [dict(experiment_id=rows[0]["experiment_id"], launched=True),
               dict(experiment_id=long["experiment_id"], launched=True)]
    with pytest.raises(ValueError):
        runner.prepare_restore_source(restore, rows, results, "77", 4)  # absent checkpoint
    fake_checkpoint(folder)
    assert runner.prepare_restore_source(restore, rows, results, "77", 4) == folder
    assert not (folder / "data.pt").stat().st_mode & 0o222  # read-only
    manifest = runner.checkpoint_manifest(folder)
    assert len(manifest) == 13 and manifest["data.pt"]["size_bytes"] == 1


def test_release_only_deletes_this_blocks_attempt_directories(scratch, tmp_path):
    rows = block_rows()
    keep = scratch / "checkpoints" / "other_group" / "x" / "attempt-77"
    keep.mkdir(parents=True)
    for row in rows:
        fake_checkpoint(checkpoint_dir(row, "77", scratch) / "global_step_1")
    folder = checkpoint_dir(rows[1], "77", scratch) / "global_step_1"
    for path in [folder, *folder.rglob("*")]:
        path.chmod(path.stat().st_mode & ~0o222)
    block_dir = tmp_path / "block"
    block_dir.mkdir()
    runner.release_checkpoints(rows, "77", block_dir)
    released = json.loads((block_dir / "checkpoint-release.json").read_text())["released"]
    assert len(released) == 3 and keep.is_dir()
    assert all(not checkpoint_dir(r, "77", scratch).exists() for r in rows)
