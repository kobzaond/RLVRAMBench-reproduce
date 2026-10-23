# Submission data dictionary

Version: `electronics-submission-2026-10-23-r2`.
The original [data dictionary](DATA_DICTIONARY.md) documents the historical
benchmark and remains unchanged where it is bound to earlier source hashes.
This document covers the submission extensions and their relationships.
`verification/dataset-views.json` lists every viewer configuration, its
source CSV, row count, and exact column names.

## Units and experimental identity

- MiB means 1,048,576 bytes. Memory peaks are per-device maxima, not sums.
- A trial is one requested invocation; an execution is one started
  scheduler execution. Reruns and a scheduler restart make these different.
- A repeated configuration ordinarily combines three distinct frozen seed
  slots. All three must be eligible before a repeated label is assigned.
- A missing completed peak is not zero. A failed run's observed peak is
  not the peak it would have reached after successful completion.
- Pair and contrast rows reuse the contributing runs. Estimator methods
  and transfer queries reuse target configurations.

## MI250X: `amd_mi250x/`

`matrix.csv` freezes 462 slots. Join it to `results/trials.csv` by `run_id`.
The latter retains the original settings plus collected outcomes.

| Field | Meaning |
|---|---|
| `panel` | `hardware_transfer`, `larger_models`, `reservation_sweep`, `algorithm_adaptation`, or `horizon` |
| `configuration_key` | Identity of the repeated settings within the panel |
| `source_configuration_id` | Corresponding A100 configuration where applicable |
| `algorithm`, `adaptation` | GRPO/PPO and LoRA/full-parameter condition |
| `gpu_count` | Number of allocated MI250X GCDs, not the number of dual-GCD packages |
| `actor_micro_batch` | Per-device actor update micro-batch |
| `vllm_gpu_memory_utilization` | Requested generation-engine memory fraction |
| `parameter_offload`, `free_cache_engine` | Actor parameter offload and generation-state release switches |
| `total_training_steps` | Requested update count |
| `job_id` | Final submitted job used for this slot's outcome |
| `attempt_job_ids` | Semicolon-separated submitted job IDs associated with the slot |
| `state` | `within_margin`, `above_margin`, `memory_failure`, or ineligible `other_failure`/`missing` |
| `failure_mode` | Diagnostic operation or evidence/infrastructure failure; blank if inapplicable |
| `completed_requested_work` | Final requested step printed and successful exit; eligibility additionally requires valid traces |
| `completed_steps`, `requested_steps`, `exit_code` | Completion and exit evidence |
| `device_capacity_mib`, `margin_limit_mib` | Measured per-GCD capacity and capacity minus 2,048 MiB |
| `peak_mib` | Maximum observed per-device memory during this attempt |
| `peak_<stage>_mib` | Observed stage maximum; blank when that stage was not sampled |
| `time_<stage>_s` | Mean recorded trainer time for that stage; not an end-to-end throughput result |
| `elapsed_seconds` | Invocation elapsed time from the terminal record |

`results/configurations.csv` groups the trials by panel, configuration,
algorithm, and adaptation. `observed_state` is the repeated label;
`planned_processes` and `eligible_processes` show its coverage.
`memory_failure_processes` preserves adverse observations even when the
repeated label is unresolved. Completed-peak summaries use completions only.
The Phi-4-mini/GSM8K c5 replay has two memory failures and one ineligible
run, so its repeated label is `unresolved`.

`results/attempt_ledger.csv` is scheduler accounting, not an additional
outcome dataset. It includes unstarted/cancelled submissions and pilots.
Its 816 rows must not be described as 816 completed trials. The formal
campaign has 510 started job IDs and 511 executions, plus 12 pilot
executions reported separately. Read its start/state/panel fields and the
four runtime-amendment records before aggregating costs.

`results/analysis/` contains comparisons with the matching A100 tables.
In `hardware_transfer_labels.csv`, unresolved rows remain part of the
90-configuration inventory; agreement uses only the 89 resolved pairs.
The horizon pair table compares distinct one-step and 40-step invocations.

## A100 follow-ups: `profiles/electronics_revision/`

`adaptation/`, `reservation/`, `release/`, and `timing/` each provide
`outcomes.csv`, cases, contrasts, and a summary. Protocols and seed matrices
are under `benchmark/electronics_revision_20261016/`.
Development runs in `pilot/` do not replace formal invalid slots.

Read each outcome's validation status before using a peak or cost. Joint
completion is required for continuous paired contrasts. The adaptation
panel records actual trainable-parameter inventories; LoRA rank zero denotes
the full-parameter condition, not distributed process rank zero.
The iteration-cost panel measures 20 updates and actual token counts.
Host-memory fields describe the measured cgroup and may include file cache;
they are not measurements of actor-owned host memory alone.

`llmem_native/pairs.csv` contains 24 prospective prediction/measurement
pairs in LLMem's native OPT/Gemini setting. It is a different workload and
runtime from the colocated GRPO trials. Unsupported colocated targets are
not silently treated as rejected predictions.

## Independent A100 offload controls

`benchmark/offload_factorial/results/slots.csv` contains 144 invocations.
`configurations.csv` has 48 three-seed configurations.
`offload_pairs.csv` has 72 matched contrasts; completed-peak contrasts are
absent when either member fails. `prediction_scores.csv` contains 42
method/stratum score rows, not 42 experiments. Predictions were frozen
before execution in the adjacent protocol directory.

## Estimator comparison

`estimator_comparison/` retains recorded DeepSpeed calls, architecture
metadata, predictions, and scores. The 70 rows in
`results/score_comparison_all_methods.csv` are method/track summaries.
All methods reuse the benchmark's 400 queries and 94 distinct target
configurations. The comparison separates native model-state accounting,
additional reservation/activation terms, and empirical references.
Applying training-state estimators to colocated RL is outside their native
scope; these rows are not evidence of estimator accuracy on every RL stack.

## Frozen evidence and historical snapshots

`release.json` and `MANIFEST.sha256` identify the submission files and the
four raw-evidence archives. Every archive has its own evidence inventory
and SHA-256 sidecar. The old `electronics_revision_2026_10/` directory and
earlier release metadata are preserved for history and are not alternate
submission targets. Use the root-level paths exposed by the current card.
