"""Execute the real payload launcher with stub cluster commands (no submission)."""
import shutil
import subprocess

import pytest

from ieee_resub import common as C
from ieee_resub import plan, sandbox
from ieee_resub.assess import TRIAL_FIELDS, normalized
from ieee_resub.render import config_keys, private_tmp_issues


@pytest.fixture(scope="module")
def paths():
    root = sandbox.new_root()
    yield sandbox.prepare(root)
    shutil.rmtree(root, ignore_errors=True)


def overrides(result):
    return dict(a.split("=", 1) for a in result["trainer_command"] if "=" in a and not a.startswith("-"))


def check_trial(result, row):
    assert result["returncode"] == 0, result["stderr_tail"]
    for field in TRIAL_FIELDS:
        assert normalized(result["trial"][field]) == normalized(row[field]), field


def test_e1_row_enables_fixed_length_canonical_order_and_shape_trace(paths):
    rows, _ = plan.e1_rows()
    row = next(r for r in rows if r["condition"] == "b16-full" and r["dataset"] == "gsm8k")
    result = sandbox.render(row, paths)
    check_trial(result, row)
    hydra = overrides(result)
    assert hydra["actor_rollout_ref.rollout.fixed_response_length"] == "True"
    assert hydra["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == "16"
    assert hydra["trainer.resume_mode"] == "disable" and hydra["trainer.resume_from_path"] == "null"
    options = " ".join(result["apptainer_options"])
    assert "RLVRAM_CANONICAL_BATCH_ORDER=1" in options and "RLVRAM_SHAPE_TRACE=/" in options
    assert "VERL_ALLOCATOR_TRACE_DIR=/" in options
    assert str(C.REPO_ROOT / C.INSTRUMENTED_PACKAGES) in options
    has = config_keys()
    assert all(has(key) for key in hydra)


def test_external_row_disables_allocator_trace(paths):
    rows, _ = plan.e1_rows()
    row = next(r for r in rows if r["condition"] == "b08-ext")
    result = sandbox.render(row, paths)
    check_trial(result, row)
    assert "VERL_ALLOCATOR_TRACE_DIR= " in " ".join(result["apptainer_options"]) + " "


def test_e2_four_gpu_row_and_e3_restore_row(paths):
    rows, _ = plan.e2_rows()
    row = next(r for r in rows if r["condition"] == "q7-gsm8k-off0-r60")
    check_trial(sandbox.render(row, paths), row)
    rows, _ = plan.e3_rows()
    row = next(r for r in rows if r["role"] == "restore")
    result = sandbox.render(row, paths)
    check_trial(result, row)
    hydra = overrides(result)
    assert hydra["trainer.resume_mode"] == "resume_path"
    assert hydra["trainer.resume_from_path"].endswith("-long/attempt-9000001/global_step_50")
    assert hydra["trainer.del_local_ckpt_after_load"] == "False"
    assert "/checkpoints/" in hydra["trainer.default_local_dir"]
    assert hydra["trainer.val_before_train"] == "False" and hydra["trainer.total_epochs"] == "100"


def test_launcher_rejects_implicit_resume_and_missing_checkpoint(paths):
    rows, _ = plan.e3_rows()
    row = dict(next(r for r in rows if r["role"] == "long"), resume_mode="auto")
    with pytest.raises(ValueError):
        sandbox.render(row, paths)  # the environment builder refuses implicit resume
    restore = next(r for r in rows if r["role"] == "restore")
    result = sandbox.render(restore, paths, job="9000005", create_checkpoint=False)
    assert result["returncode"] != 0 and "trainer_command" not in result  # launcher stops before apptainer


def private_tmp(result):
    options = result["apptainer_options"]
    binds = [options[i + 1] for i, o in enumerate(options[:-1]) if o == "--bind"]
    return [b for b in binds if b.endswith(":/tmp")]


def test_every_container_gets_this_invocations_private_tmp(paths):
    """e3-amendment-01, defect 1: no shared host /tmp inside the container."""
    rows = plan.e1_rows()[0][:1] + plan.e3_rows()[0][:3]  # E3 block 0: screen, long, restore
    for number, row in enumerate(rows):
        job = str(9000100 + number)
        result = sandbox.render(row, paths, job=job)
        assert result["returncode"] == 0, result["stderr_tail"]
        runtime = paths["scratch"] / "r" / job / f"matched-{row['experiment_id']}-j{job}"
        assert private_tmp(result) == [f"{runtime}/tmp:/tmp"]
        assert "TMPDIR=/r/tmp" in result["apptainer_options"]
        assert not runtime.exists()  # removed with the rest of the runtime
        assert private_tmp_issues(result, row, paths, job=job) == []  # token visible at the bound source


def test_in_container_check_stops_unless_tmp_is_the_private_directory(paths, tmp_path):
    """The wrapper that starts the trainer, run with /tmp emulated by a directory."""
    row = plan.e3_rows()[0][0]
    result = sandbox.render(row, paths, job="9000200")
    assert result["returncode"] == 0, result["stderr_tail"]
    command = result["trainer_command"]
    assert command[:2] == ["bash", "-c"] and command[3] == "rlvram-private-tmp-check"
    assert command[4:7] == ["python", "-m", "memory_tuner.device_contract"]
    token = next(o.split("=", 1)[1] for o in result["apptainer_options"] if o.startswith("RLVRAM_PRIVATE_TMP_TOKEN="))
    assert result["private_tmp_marker"] == token and "9000200" in token
    private, shared = tmp_path / "private", tmp_path / "shared"
    private.mkdir()
    shared.mkdir()
    (private / ".rlvram-private-tmp").write_text(token + "\n")
    (shared / ".rlvram-private-tmp").write_text("token-of-another-invocation\n")

    def inside(tmp):
        script = command[2].replace("/tmp/", f"{tmp}/")
        return subprocess.run(["bash", "-c", script, "check", "echo", "trainer-started"], capture_output=True,
                              text=True, env=dict(RLVRAM_PRIVATE_TMP_TOKEN=token, PATH="/usr/bin:/bin"))
    ok = inside(private)
    assert ok.returncode == 0 and ok.stdout.split() == ["RLVRAM_PRIVATE_TMP", "verified", "trainer-started"]
    for tmp in (shared, tmp_path / "absent"):
        stopped = inside(tmp)
        assert stopped.returncode == 97 and "trainer-started" not in stopped.stdout
        assert "RLVRAM_PRIVATE_TMP failed" in stopped.stderr


def test_amendment_rerun_resumes_from_the_preserved_checkpoint_only(paths):
    rows, _ = plan.e3_rows()
    row = next(r for r in rows if r["experiment_id"] == "e3-m7-code_heavy_tail-off1-r70-s631-restore")
    preserved = "e3-m7-code_heavy_tail-off1-r70-s631-long-attempt-5381357/global_step_50"
    result = sandbox.render(row, paths, job="9000002", preserved_source=preserved)
    check_trial(result, row)
    hydra = overrides(result)
    assert hydra["trainer.resume_from_path"] == str(paths["scratch"] / "preserved-checkpoints" / preserved)
    assert hydra["trainer.default_local_dir"].endswith(f"{row['experiment_id']}/attempt-9000002")
    elsewhere = sandbox.render(row, paths, job="9000003", preserved_source="../outside/global_step_50")
    assert elsewhere["returncode"] != 0 and "trainer_command" not in elsewhere
