# RLVRAMBench data dictionary

Protocol 1.0. CSV files use UTF-8, a header row, and a blank cell for missing
or unavailable information. Blank is **not zero**, success, or a memory
failure. Measurements in MiB use 1 MiB = 1,048,576 bytes. Identifiers are
opaque stable keys, not numerical features. All listed paths are relative
to the extracted evidence archive.

## Tables and joins

| File in `benchmark/` | Unit | Key / relation |
|---|---|---|
| `transfer_configurations.csv` | One of 98 configurations used as a source or target | `configuration_id`; convenient settings/outcomes join for browsing |
| `configurations.csv` | One fixed setting within a study (212 rows) | `configuration_id`; no measured outcomes |
| `outcomes.csv` | Eligible and all-attempt summary for that setting | One-to-one with `configurations.csv` |
| `configuration_results.csv` | All 212 settings joined to outcomes | Browsing only, not blind candidate input |
| `runs.csv` | An eligible process (612 rows) | `run_id`; belongs to one configuration |
| `attempts.csv` | A recorded invocation (683 rows) | `attempt_id`; eligible invocations link to `run_id` |
| `stage_measurements.csv` | Externally sampled peak in an observed execution stage | (`run_id`, `stage`); 3,434 rows |
| `queries.csv` | A source-to-target prediction request (400 rows) | `query_id`; grouped by `task_id` |
| `not_started_conditions.csv` | A planned control condition that never started (6 rows) | Legacy condition/allocation record, not an attempt |

The 612 eligible runs comprise the paper's original 588-process corpus and
24 completed batch-by-logging controls. The 683 attempts comprise 657
original attempts and 26 started controls. Sixteen historical source screens
remain in the separate archival table
`profiles/standard_grpo/historical-temporal-source-trials.csv`; they are
not added to these totals. The archived `factorial-phase-trials.csv` is a
different view of existing runs, not another experiment.

## Configuration and case identity

`configuration_id` hashes the study, model, workload, condition, and explicit
settings below, excluding outcomes, seeds, job IDs, and source revisions.
`case_id` groups one study, model, workload, GPU count, and requested horizon.
Keeping this grouping does not assert that historical runs used one
immutable working copy. Raw environment and source records preserve that
separate provenance limitation.

- `study`: boundary, state release, batch/reservation, original 40- or
  100-step extension, sampling calibration, instrumentation control,
  horizon follow-up, four-GPU transfer, or batch-by-logging control.
- `model_family`: legacy short identifier; `model_name` is the readable
  checkpoint name; `model_id` identifies its upstream repository.
- `workload_id`, `workload_name`: stable identifier and readable name.
  `legacy_dataset_id` retains the original table spelling. In particular,
  `code_heavy_tail` is displayed as **Longer-prompt code**; this name does
  not assert a measured heavy-tailed distribution.
- `algorithm`: Group Relative Policy Optimization (GRPO).
  `adaptation`: low-rank adaptation (LoRA), training adapter matrices.
- `configuration_level`: test index or named experimental setting,
  **not a throughput/quality rank**. `c0`–`c5` jointly change the actor
  micro-batch, generation-memory reservation, and actor parameter
  offload (enabled at c0–c1, disabled at c2–c5). Log-probability
  micro-batches remain fixed by workload: 8 for MATH and 4 otherwise.
  Actual values are in adjacent columns; do not infer
  them from the index. Control settings and interpolated horizon settings
  are not additional levels of the six-level grid.
- `condition`: experimental treatment, such as release/resident,
  external/full instrumentation, source/long horizon, or the batch/logging
  combination. Blank means that no separate condition field was recorded.
- `used_in_transfer_tasks`: 1 for the 98 scored-source/target configurations,
  0 for supporting measurements. Supporting target-case controls are not
  permitted extra information for transfer prediction.

## Explicit resource and workload settings

| Column | Meaning and unit |
|---|---|
| `gpu_count` | Number of allocated graphics processing units |
| `accelerator`, `device_capacity_mib` | Device model and reported capacity per device |
| `margin_limit_mib` | Chosen per-device limit, 38,912 MiB, inclusive |
| `peak_measurement` | Measurement convention, maximum externally sampled per-device memory |
| `rollout_tp_size` | Number of devices sharing each generation model by tensor parallelism |
| `actor_micro_batch` | Training examples processed together per actor micro-batch |
| `rollout_logprob_micro_batch` | Generation-policy log-probability micro-batch size |
| `ref_logprob_micro_batch` | Fixed-reference-policy log-probability micro-batch size |
| `vllm_gpu_memory_utilization` | Fraction reserved by the vLLM generation engine; not measured utilization |
| `parameter_offload` | Whether model parameters are offloaded from device memory |
| `optimizer_offload` | Whether optimizer state is offloaded |
| `free_cache_engine` | Whether generation state is released between stages |
| `max_prompt_length` | Maximum prompt tokens under the selected model's tokenizer |
| `max_response_length` | Maximum generated response tokens |
| `max_model_len` | Configured maximum total model sequence length, tokens |
| `max_num_seqs` | Maximum concurrent sequences configured for generation |
| `rollout_n` | Generated responses per prompt |
| `train_batch_size` | Global training batch size in prompts |
| `train_max_samples`, `val_max_samples` | Preparation/sample limits; not actual processed counts |
| `total_training_steps` | Requested training horizon |
| `gpu_monitor_interval_ms` | Requested external device-memory sampling interval, milliseconds |
| `allocator_trace_enabled` | Whether additional allocator instrumentation is enabled |
| `allocator_trace_sync` | Whether allocator instrumentation waits for pending GPU work |
| `save_freq`, `test_freq` | Configured checkpoint/validation intervals in steps; negative disables periodic events |
| `val_before_train` | Whether validation precedes training |
| `resume_mode` | Recorded checkpoint-resume setting |
| `max_actor_ckpt_to_keep` | Checkpoint-retention setting |

Boolean settings retain `True`/`False` from the frozen matrices. A blank
setting means it was not present in that source table, not that the option
was disabled. Raw configuration and environment records remain authoritative
for defaults. A final-step event may occur in addition to periodic events.

The measurements used the Karolina supercomputer at IT4Innovations in
Ostrava, Czech Republic. They cover a particular colocated VERL/vLLM
implementation on A100 devices; these columns do not specify a portable
replacement for that execution stack.

## Outcomes

The three output labels apply to **three distinct eligible seed slots**:

- `within_margin`: all three complete, each at or below 38,912 MiB.
- `above_margin`: all three complete, but at least one exceeds that limit.
- `memory_failure`: at least one eligible process has a diagnosed memory
  failure. Its sampled peak need not reach capacity before the failure.

`not_repeated` appears only for the twelve single-process sampling
calibration settings. It is not a fourth prediction label; those settings
never enter the transfer tasks.

`eligible_processes`, `distinct_seeds`, `completed_eligible_processes`,
`within_margin_processes`, `above_margin_processes`, and
`memory_failure_processes` are counts, not probabilities.
`max_completed_run_peak_mib` and `mean_completed_run_peak_mib` use only
completed eligible processes. They are blank if none complete and must not
be mistaken for the unobserved peak a failed process would have reached.
`recorded_attempts`, `documented_completed_attempts`, and `excluded_attempts`
describe the full recorded history. `label_scope` identifies eligible
resource outcomes or single-process calibration.

A slot can receive an eligible replacement after a documented exclusion.
This does not erase the earlier attempt or turn its cause into an
out-of-memory diagnosis. For the original 100-step extension, twelve
attempts complete out of fourteen, whereas all twelve selected eligible
outcomes complete within margin. Two excluded attempts stop after 63 and
77 steps; one reports a CUDA invalid argument and the other has unresolved
weight-synchronization nonprogress.

## Runs, attempts, and stages

- `experiment_id` identifies a planned configuration/seed slot; `job_id`
  identifies its scheduler allocation. Several processes can share an
  allocation, so a job ID alone is not a run ID.
- `training_seed`, `pair_id`, `period` preserve random-seed and paired-order
  information. Do not split those repeats into training/test examples.
- Run `observed_state` is the single-process resource result;
  `completed_requested_work` is 0/1 from the validated analysis, not from
  the legacy stage counter alone. Some completed calibration logs have
  a stage counter stuck at zero.
- `requested_training_steps`, `elapsed_seconds`, `observed_peak_mib`,
  and `completed_run_peak_mib` give the requested horizon, elapsed seconds,
  sampled peak, and peak only if the run completed.
- `terminal_stage` identifies the last observed stage; `failure_kind`
  retains the diagnosed failure category. A blank failure kind is not
  evidence that an excluded attempt succeeded or had a known non-memory cause.
- `source_table` names the source CSV. `artifact_path`, `run_log`,
  `phase_memory_csv`, `gpu_telemetry_csv`, `environment_json`, and
  `allocation_manifest` locate the corresponding archive evidence.
  `source_head` is the source revision when provided by the source table.
- Attempt `eligible` is 0/1; `recorded_outcome` and `exclusion_reason`
  retain the audit's wording. `documented_completion=0` means completion is
  not established, not a newly diagnosed failure. `evidence_files` is a
  semicolon-separated list and `scheduler_evidence` points to additional
  scheduler evidence when available.
- `last_completed_step_if_documented` is populated for the two late
  excluded 100-step attempts from their execution logs; blank elsewhere
  means not curated here, not step zero.
- Stage `stage` is initializing, rollout (generation), reference_logprob
  (reference-policy scoring), actor_update (training), weight_sync
  (weight transfer), validation, checkpoint, idle, or unknown.
- Stage `coverage` is `completed_run`, `terminal_stage_partial`, or
  `stage_completion_not_certified`. The last two distinguish failure
  records from complete-run measurements. They do not certify completion
  of an earlier stage; consult the traces for a stage-specific contrast.
  All stage peaks here are **external device-memory measurements**, not
  PyTorch allocated-memory peaks.

The six never-started rows retain the control inventory's fields:
`experiment_id`, `pair_id`, `condition`, `period`, `model_family`, `dataset`,
`training_seed`, `job_id`, `initial_allocation`, `attempted`, `outcome`,
`observed_peak_mib`, `actor_update_peak_mib`, and `terminal_sampled_phase`.
`attempted=0` means they are not included in any attempt denominator.

## Queries, predictions, and metrics

`query_id` is the prediction key. `task_id` groups requests that share an
allowed donor case; `track` gives the transfer type. `source_configuration_id`
and `target_configuration_id` join to settings and outcomes;
`source_case_id` and `target_case_id` make the separation explicit.
Each row is a comparison, **not a new experimental configuration**.

Prediction CSVs have exactly one row per selected query and columns
`query_id,predicted_state`. The evaluator rejects missing, duplicate,
unexpected, or invalid predictions. Approval means predicting
`within_margin`; the other two labels reject the target.

The score JSON reports correct three-state counts/accuracy, distinct target
counts, approvals, approved memory failures, approved margin violations,
and rejected within-margin targets. Each error is shown against its
relevant target-state count; approval errors are also shown against the
approved count. The confusion matrix has **predicted rows and observed
columns**. A rate with zero denominator is JSON `null`, never an artificial
zero error. Per-task and per-track results accompany descriptive pooled
counts. Macro task accuracy weights tasks equally within a track.

Source costs count distinct source configurations, eligible processes,
and recorded attempts within each reported scope. They are not GPU-hours
or the cost of acquiring any extra measurements. Shared donors mean that
task costs must not be summed to estimate the whole release's cost.
