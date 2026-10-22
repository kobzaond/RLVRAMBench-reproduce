# Cross-accelerator replication on AMD MI250X (LUMI)

This panel replays RLVRAMBench configurations on AMD Instinct MI250X
accelerators of the LUMI supercomputer. It tests which A100 observations
survive a change of accelerator **and** software stack. It does not isolate
a hardware effect.

| | A100 study | This panel |
|---|---|---|
| Device | A100-SXM4-40GB, 40,960 MiB | MI250X GCD, 65,520 MiB |
| Operating limit | 38,912 MiB | 63,472 MiB (capacity − 2,048) |
| Runtime | CUDA 12.9, PyTorch 2.10.0 | ROCm 7.2.3, PyTorch 2.11.0 |
| vLLM / VERL | 0.18.0 / instrumented development checkout | 0.25.1 / 0.9.0 release |
| Memory sampler | NVML, 100 ms | amdgpu sysfs `mem_info_vram_used`, 100 ms |
| Allocator traces | yes | no |

The VERL settings are those of the A100 runs: LoRA rank 64 and alpha 32,
KL loss, the same micro-batches, generation budgets, offload switches,
length limits, and seeds.

## Files

| File | Role |
|---|---|
| `plan_matrix.py` → `matrix.csv` | Frozen 462-invocation matrix (five panels, see below) |
| `run_grpo_mi250x.slurm` | One invocation: monitor, VERL launch, trial record |
| `gpu_monitor_amd.py` | sysfs sampler; same CSV layout as `memory_tuner/gpu_monitor.py` |
| `hook/sitecustomize.py` | Stage-marker writer and two compatibility repairs (below) |
| `prepare_data.py` | GSM8K / MATH parquet files in VERL's stock format |
| `submit.py` | Submits each matrix row at most once, plus amendment reruns |
| `collect.py` → `results/trials.csv`, `configurations.csv`, `summary.json` | Outcomes from raw logs and samples |
| `analyze.py` → `results/analysis/` | Comparison with the A100 tables in `../benchmark` |
| `runtime_amendment_0[1-3].json` | Harness defects and the slots rerun because of them |
| `venv-freeze.txt` | Exact Python packages of the project venv |

## Panels

1. `hardware_transfer`: every A100 GSM8K/MATH configuration of the boundary,
   batch/reservation, four-GPU and state-release studies, with the A100
   seeds (90 configurations, 270 runs).
2. `larger_models`: GSM8K/MATH part of the 7B completion replication
   (Qwen2.5-7B, Mistral-7B; c1–c3; 2 and 4 GCDs; 72 runs).
3. `reservation_sweep`: vLLM reservation 0.30–0.95 at fixed c3 settings,
   Qwen-3B and Phi-4-mini on GSM8K (48 runs).
4. `algorithm_adaptation`: GRPO-LoRA, PPO (GAE, full-parameter critic) and
   full-parameter GRPO at c2/c3, Qwen-1.5B and Qwen-3B on GSM8K (36 runs).
5. `horizon`: one-step screens and separate 40-step runs of the A100
   temporal configurations (36 runs).

Code workloads are not replicated: their A100 preparation and reward are
not part of this repository.

## Hook patches

`hook/sitecustomize.py` is on `PYTHONPATH` in every container process.

* Wraps `verl.utils.debug.marked_timer` so VERL's own stage timers also
  write `<unix_ns>,<step>,<stage>` to the marker file read by the sampler.
* Loads in-memory LoRA tensors with vLLM's *unstacked* weight mapper, as
  vLLM's own file-based loader does. Without it, packed q/k/v and gate/up
  layers receive one fused tensor and `set_lora` fails on the first sync.
* Makes the agent loop's chat-template probe tolerate Mistral's
  alternating-role check. The probe only finds tokens needed by
  multi-turn tool loops; single-turn prompts are rendered unchanged.

None of these changes a memory setting.

## Outcome rules

As in `BENCHMARK.md`:

* A run completes only when the console shows the final requested step.
* A memory failure needs log evidence.
* `other_failure` (no completion and no memory diagnosis) is ineligible.
* A configuration label needs three eligible runs; any memory failure makes it `memory_failure`.

Failure modes come from the traceback around the first device-OOM message:

* `rollout_startup_admission`: vLLM refuses its budget at start-up.
* `generation_wakeup_oom`: vLLM sleep-mode `wake_up` re-allocation, which on this stack runs inside VERL's weight-update interval.
* `update_weights_oom`
* `update_actor_oom`
* `gen_oom`

## Runtime amendments

All amendments rerun slots with logged evidence of a harness defect. Every
attempt keeps its job-ID-named files and its line in `submissions.csv`.
Diagnosed memory outcomes are never rerun.

1. **Shared `/tmp`.**
   * The defect: VERL names its weight-transfer ZMQ socket `ipc:///tmp/rl-colocate-zmq-<ray job id>-…`. Every fresh Ray cluster reuses the same first job ID, so co-located jobs connected to each other's sockets. Receivers got `hipErrorInvalidDevicePointer` or hung.
   * Fix: a private `/tmp` per job.
2. **Missing `psm_*` segments and port races.**
   * Missing `psm_*`: vLLM's shared-memory segments in the node-wide `/dev/shm` vanished at start-up. The cause was not established. Singularity refuses to bind a directory onto `/dev`, so no fix is applied.
   * Port races: vLLM lost ephemeral-port races with co-located jobs.
   * Also rerun under this amendment: jobs that had already started under the shared-`/tmp` harness.
3. **Faulty amendment-02 port setting.**
   * The defect: amendment 02's job-specific `VLLM_PORT` made a job's two vLLM servers race each other for one port. All 17 amendment-02 reruns failed.
   * Fix: the setting was removed and those slots ran once more.
   * Provenance: the recorded `harness_sha256` now hashes Slurm's executing copy of the batch script (`$0`), not the checkout.

One vLLM sleep-mode self-check failure ("Memory usage increased after
sleeping") showed none of these defects and keeps its non-memory outcome.

## Reproduce

Paths are for LUMI project 465003040. Adjust `AMD_ROOT` in the scripts for
another allocation.

```bash
# container + venv (inside the container, --system-site-packages)
python3 -m venv --system-site-packages $AMD_ROOT/venv
$AMD_ROOT/venv/bin/pip install -r amd_mi250x/venv-freeze.txt   # verl 0.9.0, transformers 5.10.2, ...
python3.11 amd_mi250x/prepare_data.py --output $AMD_ROOT/data
python3.11 amd_mi250x/plan_matrix.py
python3.11 amd_mi250x/submit.py                      # repeat until all rows are submitted
python3.11 amd_mi250x/collect.py && python3.11 amd_mi250x/analyze.py
```

`collect.py` reads the raw run directories and asks `sacct` about jobs
without a trial record. `analyze.py` needs only `results/` and `../benchmark`.
