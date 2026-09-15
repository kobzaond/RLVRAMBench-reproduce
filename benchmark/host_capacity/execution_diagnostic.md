# Host-capacity diagnostic and final collection — 2026-09-15

All nine array elements of `5055845` ended `FAILED 2:0`, with no scheduler `OUT_OF_MEMORY` state. The last element, index 8 / actual job `5055845`, ended at **19:27:18 UTC**.

Final reconstruction: **18 launches; 9 eligible 2-GPU rollout-initialization memory failures; 9 unresolved 4-GPU periods; zero eligible completions.** Three of six configurations resolve, all 2-GPU failures. Evaluation is restricted to that resolved subset. No eligibility, predictor, runtime, or raw-evidence changes were made; no experiments or retries were launched.

## Diagnosis and limits

The evidence does **not** support “no training was performed.” In every 4-GPU period, all four ranks recorded an `actor_update` call returning without an exception, followed by a separate post-update `weight_sync` call returning without an exception. The recorded phase sequence reaches step-1 actor update, weight synchronization, then idle. These are completed calls, not independent proof that optimizer parameters changed or that the entire required terminal protocol completed.

All nine 4-GPU dispatches specify `TOTAL_TRAINING_STEPS=1`, `TRAINER_LOGGER=["console"]`, and `RESUME_MODE=disable`. The examined resolved trainer configuration agrees; validation and checkpointing are disabled. Every 4-GPU payload exited `0`, but every period has:

```text
status: infrastructure_invalid
validation_error: ValueError: Successful exit lacks the exact one-step terminal event
```

Neither a `step:` metric line nor `training/global_step:` appears in any of the nine retained training logs. All nine post-period allocation-idle checks passed.

Pinned source root:

```text
/mnt/proj3/open-35-44/verl/rl/job_sources/host-capacity-120e29a
commit: 120e29a146294722a6668f55a1e7fed2878eb62a
```

Relevant control flow, relative to that root:

- `sppo_replay/allocator_trace.py`: the decorator records `error_type` in `finally`, catches and rethrows `BaseException`, and does not suppress failures. All 36 actor-update and 36 subsequent synchronization records examined have empty `error_type`.
- `instrumented_python_packages/verl/workers/engine_workers.py:719`: the actor-update decorator encloses `update_actor`, which calls `self.actor.train_mini_batch`. This is the training-call path, not merely a phase label.
- `instrumented_python_packages/verl/trainer/ppo/v1/trainer_sync.py:35`: `on_step_end` performs weight synchronization before marking idle.
- `instrumented_python_packages/verl/trainer/ppo/v1/trainer_base.py:497`: after synchronization, the normal path computes metrics, logs them at line 508, updates progress at line 513, and returns at line 521. The examined post-update path has no normal return bypassing that logging.
- `instrumented_python_packages/verl/utils/logger/aggregate_logger.py:49`: the console logger prints with `flush=True`. A CPU-only check of this exact formatter emitted `step:1 - training/global_step:1`; the frozen terminal regex recognized it. This is not a demonstrated formatting/regex mismatch.
- `instrumented_python_packages/verl/trainer/main_ppo.py:93` waits for the remote trainer with `ray.get`; `memory_tuner/device_contract.py:254` then returns from `runpy` into an immediate explicit `ray.shutdown()` at line 257.

The pinned container is `/mnt/proj3/open-35-44/verl/images/verl-vllm018-dev1.sif`, SHA-256 `3dc581ca780cd19d565445b8a02e5daa98b350b3d2116d4d64476a64a2614912`. CPU-only source inspection inside it confirmed Ray **2.54.1**:

- `/usr/local/lib/python3.12/dist-packages/ray/_private/worker.py:2128` waits 0.5 seconds for log delivery only during interpreter-exit shutdown. Explicit shutdown skips this grace.
- `worker.py:2788` stops the logging threads and closes the log subscriber. Flushing already received deduplicated messages does not drain upstream actor logs.
- `_private/log_monitor.py:471` forwards worker log files asynchronously; its idle loop sleeps 0.1 seconds.

**Immediate explicit shutdown therefore supplies a concrete final-log-loss mechanism and is the strongest explanation, not proof of the missing line's emission.** Original Ray actor stdout was not retained; the launcher removes its scratch runtime. `trainer-terminal.json` records driver teardown, not a completed training step. The final squashfuse background-cleanup messages are not terminal training evidence. All nine 4-GPU periods remain unresolved under the unchanged gate.

## All nine affected 4-GPU periods

Times below are the exact latest `finished_ns` across ranks 0–3, rendered as UTC on 2026-09-15. “Sync” means the synchronization **after** the actor update, excluding initialization synchronization. Peaks are diagnostic raw trial peaks, not eligible completed-run peaks.

| Index | Workload / seed | Actual job / period | Actor update finished | Post-update sync finished | Raw peak MiB |
| --- | --- | --- | --- | --- | ---: |
| 0 | code_heavy_tail / 161 | 5055846 / 1 | 19:01:31.699704267 | 19:02:18.386028932 | 39842 |
| 1 | gsm8k / 163 | 5055847 / 2 | 19:05:57.381348829 | 19:08:02.327763208 | 38560 |
| 2 | code_heavy_tail / 162 | 5055848 / 2 | 19:04:48.459750884 | 19:05:59.536233566 | 39392 |
| 3 | gsm8k / 162 | 5055858 / 1 | 19:09:58.895325061 | 19:10:32.865654877 | 38560 |
| 4 | math / 162 | 5055859 / 2 | 19:14:44.017014711 | 19:15:55.046759828 | 39086 |
| 5 | code_heavy_tail / 163 | 5055860 / 2 | 19:16:12.444318056 | 19:17:22.263802288 | 39648 |
| 6 | math / 163 | 5055863 / 1 | 19:19:05.722544967 | 19:20:47.573877478 | 39086 |
| 7 | math / 161 | 5055864 / 1 | 19:21:44.142735712 | 19:22:20.560774115 | 39086 |
| 8 | gsm8k / 161 | 5055845 / 2 | 19:25:33.619484057 | 19:27:04.665561977 | 38560 |

Exact evidence locator:

```text
BASE=/mnt/proj3/open-35-44/verl/rl/output/estimation_host_capacity
PAIR=host-qwen25-7b-<workload>-c2-s<seed>
ATTEMPT=BASE/PAIR-4gpu/attempt-<actual-job>

ATTEMPT/allocator-traces/allocator-<pid>.jsonl
ATTEMPT/device-evidence/worker-actor_rollout-<rank>-<pid>.json
ATTEMPT/training-<actual-job>.log
ATTEMPT/trial-<actual-job>.json
BASE/pairs/PAIR/period-<period>.json
```

Worker PIDs in rank order 0, 1, 2, 3:

```text
5055846: 758712,758713,758714,758715
5055847: 3563810,3563811,3563812,3563813
5055848: 3495668,3495669,3495670,3495671
5055858: 802123,802124,802125,802126
5055859: 3539673,3539674,3539675,3539676
5055860: 3613579,3613580,3613582,3613583
5055863: 843954,843955,843956,843957
5055864: 3571156,3571157,3571158,3571159
5055845: 3657525,3657526,3657527,3657528
```

Each allocator file was checked against its period-record hash and size; its PID/rank was matched to worker identity evidence. Each contains exactly one actor-update record and one subsequent synchronization record, with ordered timestamps and empty errors.

## Accounting and copy paths

Saved terminal scheduler response:

```text
/tmp/rlvrambench-host-final-20260915.u6ynp1/sacct-raw.psv
SHA256: 5c672842d7ad1ef71e6164ae76ba2e74400c45c2511a66d4139489f34664d2eb
```

Portable, pair-claim-bound accounting input used by the collector:

```text
/tmp/rlvrambench-host-final-20260915.u6ynp1/allocation_accounting.json
SHA256: 8df8040c1cd1608abd10759dd42448d734aa1b501733c926978247d2177f12ee
```

The raw response was captured with `TZ=UTC`, `SLURM_TIME_FORMAT=standard`, and:

```text
sacct -n -P -j 5055845 --format=JobID,JobIDRaw,JobName,State,ExitCode,Start,End,ElapsedRaw,AllocTRES,NodeList,Reason
```

`memory_tuner.capture_allocation_accounting` consumed that saved response with `--root /mnt/proj3/open-35-44/verl/rl --group estimation_host_capacity`. All nine accounting records validate against the final manifests/ledgers. Independently summing outer elapsed seconds × four GPUs gives **20,828 GPU-seconds = 5.785555556 GPU-hours**. Task-provisioned invocation time is 17,605.783111 GPU-seconds; four-GPU invocation-reservation time is 20,199.935106 GPU-seconds. These overlapping scopes must not be added. Preflight cost is outside this study-array collection.

## Frozen provenance and final outputs

The mutable collector files matched Volta's integration report. Frozen source/protocol inputs were read from the immutable source root above; raw evidence was read from the canonical project. External execution authority was:

```text
/home/kobzaond/RLVRAMBench-revision-20260915/benchmark/host_capacity/execution_commits.json
```

Its exact nine-pair mapping requires commit `120e29a146294722a6668f55a1e7fed2878eb62a` throughout. Anchors verified:

```text
Host protocol: f3f49617a7fca63ddac8923bccb83a15eaf52646e5f3db5df4aa716bb25c1ef6
Host freeze:   80559199c59246d0e4ce52bf268d0375c90fc5168a5246ed900c9266abd951cd
Original prediction seal:
5ed714caff922a5109346a4242545d19f2824636d0f52bb0bbd2a09b40ed4666
```

Final output directory, exactly 13 files:

```text
/tmp/rlvrambench-host-final-20260915.u6ynp1/host-results-final
output_manifest.json SHA256:
47fe58631433062547800641c4e4a4ed5136b661db1c297d6ef24d15c1012709
```

All 12 manifest-listed output hashes/sizes and 396 raw-evidence, 20 source-input, and five protocol-input hash/size records were independently rechecked. The output includes explicit unresolved configurations and unscored approvals; no all-panel accuracy or eligible 4-GPU peak is asserted.
