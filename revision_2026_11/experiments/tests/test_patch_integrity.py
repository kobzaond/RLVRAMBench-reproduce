"""The VERL changes are a reproducible, opt-in patch; the original tree is untouched."""
import py_compile
import shutil
import subprocess

from ieee_resub import common as C

PATCH = C.EXPERIMENT_DIR / "verl_patch/ieee_resub_verl.patch"
COPY = C.REPO_ROOT / C.INSTRUMENTED_PACKAGES
CHANGED = {
    "verl/RLVRAMBench_MODIFICATIONS.md", "verl/trainer/config/_generated_ppo_trainer.yaml",
    "verl/trainer/config/rollout/rollout.yaml", "verl/trainer/constants_ppo.py",
    "verl/trainer/ppo/v1/trainer_base.py", "verl/utils/rlvram_shape_trace.py",
    "verl/workers/config/rollout.py", "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
}


def test_original_instrumented_tree_is_unmodified():
    status = subprocess.run(["git", "-C", str(C.REPO_ROOT), "status", "--porcelain", "--",
                             "instrumented_python_packages", "run_grpo_instrumented.slurm", "benchmark", "memory_tuner"],
                            capture_output=True, text=True, check=True).stdout
    assert status == ""


def test_patch_applies_to_pristine_tree_and_reproduces_copy(tmp_path):
    files = subprocess.run(["git", "-C", str(C.REPO_ROOT), "ls-files", "instrumented_python_packages"],
                           capture_output=True, text=True, check=True).stdout.split()
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(C.REPO_ROOT / name, target)
    subprocess.run(["patch", "-p1", "-s", "-i", str(PATCH)], cwd=tmp_path, check=True)
    assert C.tree_manifest(tmp_path / "instrumented_python_packages") == C.tree_manifest(COPY)
    original = C.tree_manifest(C.REPO_ROOT / "instrumented_python_packages")
    copy = C.tree_manifest(COPY)
    differing = {k for k in set(original) | set(copy) if original.get(k) != copy.get(k)}
    assert differing == CHANGED


def test_patched_python_compiles(tmp_path):
    for name in CHANGED:
        if name.endswith(".py"):
            py_compile.compile(str(COPY / name), cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_new_behaviour_is_opt_in():
    server = (COPY / "verl/workers/rollout/vllm_rollout/vllm_async_server.py").read_text()
    assert 'if self.config.get("fixed_response_length", False):' in server
    assert 'sampling_params["min_tokens"] = max_tokens' in server
    rollout = (COPY / "verl/workers/config/rollout.py").read_text()
    assert "fixed_response_length: bool = False" in rollout
    trainer = (COPY / "verl/trainer/ppo/v1/trainer_base.py").read_text()
    assert "if rlvram_shape_trace.canonical_order_enabled():" in trainer
    assert "if rlvram_shape_trace.trace_path():" in trainer
    assert 'self._mark_phase("checkpoint_load")' in trainer
