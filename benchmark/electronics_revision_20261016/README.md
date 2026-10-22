# Electronics revision evidence

These cohorts extend the historical RLVRAMBench release. They are analyzed
separately; their fresh seeds do not replace historical attempts.

| Directory | Planned formal work |
|---|---|
| `adaptation` | 96 native-KL and 12 no-KL invocations, five updates each |
| `reservation` | 144 one-update invocations at six reservation fractions |
| `release` | 36 one-update invocations around two historical anchors |
| `timing` | 24 twenty-update invocations, four conditions in each block |
| `llmem_native` | 24 prediction/measurement pairs, 48 separate processes |

The three `pilot` invocations and the two `llmem_development` scheduler
attempts are development evidence. Neither contributes to formal accuracy
or memory-contrast denominators.

Each formal directory contains a frozen matrix and protocol plus a
submission receipt anchoring the source commit and protocol hash.
`adaptation/analysis_plan.md` and `followup_analysis_plan.md` describe the
pre-execution comparisons. Later analysis and tooling notes do not amend
these execution contracts.

## Main RL fields

Matrix booleans are serialized as `True`/`False`; integers such as
`lora_rank` are CSV strings before parsing.

| Field | Meaning |
|---|---|
| `experiment_id` | One permanent planned invocation |
| `configuration_id` | A setting repeated with three seeds |
| `block_id`, `block_index` | Same-allocation comparison group and Slurm array index |
| `period` | Frozen execution position within the block |
| `training_seed` | Seed shared by conditions in the matched block |
| `lora_rank` | 64 for adapter training; 0 for full actor training |
| `use_kl_loss` | Native-KL versus separately analyzed no-reference control |
| `parameter_offload`, `optimizer_offload` | Separate actor-state placement controls |
| `free_cache_engine` | Generation-state release versus continued residency |
| `gpu_count` | GPUs used by the task, not necessarily all GPUs reserved by Slurm |
| `vllm_gpu_memory_utilization` | Engine reservation fraction, not a cap on combined colocated demand |

Applied settings are checked against the printed actor configuration.
Parameter records verify actor trainability and adaptation/reference layout;
unused critic configuration fields do not establish the active algorithm.

## Outcome and missing-data contract

`profiles/electronics_revision/PANEL/outcomes.json` preserves every planned
slot. Its CSV counterpart serializes nested values as JSON.

- `within_margin`: eligible completion with sampled maximum at most
  38,912 MiB on every task GPU.
- `above_margin`: eligible completion above that operating margin.
- `memory_failure`: eligible unsuccessful execution with an explicit GPU
  memory diagnostic.
- `unresolved`: launched or otherwise recorded work without the complete
  eligibility evidence. A successful process can be unresolved because of
  invalid telemetry.
- `not_launched`: no valid terminal launch evidence for the planned slot.

`validation_error` explains exclusion; it is not a memory-failure label.
`failure_kind` retains the older coarse classifier's terminology.
`diagnosis` adds error-and-operation-based subclasses, without claiming
allocation ownership or fragmentation.

`observed_peak_mib` can describe a failed partial execution.
`completed_peak_mib` exists only for eligible completions. Partial peaks
never enter completed-demand regressions or paired numerical contrasts.
`stage_peaks_mib` locates observations in host-side operation intervals.
`global_peak_stages` may contain ties; `actor_reaches_global_peak` does not
mean that every live allocation belongs to the actor.

A three-repeat configuration receives a label only when all planned repeats
are eligible; then the worst outcome is used. Numerical prediction targets
require all repeats to complete and use their maximum completed peak.
Unresolved targets remain visible outside scored denominators.

## Contrasts and costs

`contrasts.csv` retains unresolved and failure/completion pairs.
`resolved` concerns both categorical outcomes; `both_completed` is additionally
required for numeric peak or cost differences. Differences are alternative
minus reference, and ratios are alternative/reference:

- adaptation: full minus LoRA, with native-KL and no-KL kept separate;
- reservation: actor offload minus resident actor;
- release: resident generation state minus released state;
- timing: the named alternative minus the offload-on, 0.45, release-on anchor.

The contrast's reservation field refers to its reference condition.
Both permanent invocation identifiers permit joining the exact settings.
The adaptation panel also supplies `offload_contrasts.csv`, comparing actor
offload within each fixed adaptation mode, objective, and reservation.

Performance fields prefixed `warmup_` describe updates 1–2; `steady_`
describes later updates. The timing panel's primary interval is updates
3–20. Token rates use summed observed prompt-plus-response tokens divided
by summed step time. Finite actor losses/gradient norms and response-length
records are checked, but this does not establish learning quality.
`*_step_seconds_sum` and `*_sequences_processed` retain summed time and
realized sequence counts alongside normalized rates.

Host-memory GiB use the job's batch cgroup, with anonymous memory and charged
file cache separate. Missing host evidence is not zero. Cache from prior
processes in the allocation can remain charged; these are not actor-owned
memory measurements. CUDA admission diagnostics and NVML capacity remain
distinct recorded quantities.

`invocation_seconds` includes startup and failure time.
`task_gpu_seconds` multiplies it by task GPU count.
`resource_accounting/` separately preserves the final Slurm snapshot and
outer allocation GPU-hours. These overlapping cost views must not be added.

## Prediction and external-comparator scope

Reservation predictions were frozen before execution without refitting.
The original “low reservation” rule still means exactly 0.45. Scores retain
endpoints and intermediate fractions separately, including model, workload,
and actor-offload strata. Unsupported external targets are abstentions and
never count as correct rejections.

The native LLMem table measures OPT/Gemini training, not the full RL loop.
`pairs.csv` retains all 24 formal pairs; `cases.csv` summarizes the eight
model/batch/length cells. Prediction intervals require GPU initialization
and profiling; they exclude imports and previously cached compilation.
Training peaks and whole-process peaks are distinct measured fields.

See `paper/REPRODUCING_ELECTRONICS_REVISION.md` for reconstruction and source
archives, and `paper/electronics_measurement_audit.md` for the phase-marker
exclusion and a later fix that was not used in this frozen campaign.
