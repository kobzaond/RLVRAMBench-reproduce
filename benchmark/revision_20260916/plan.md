# Revision experiments following the second supervisor review

This revision addresses limited model coverage, unresolved larger-model
executions, and placement/order confounding. It does not promise an acceptance
probability or reinterpret the two historical larger-model panels.

## Evidence to add

1. **Full-grid batch/reservation replication.** Cross actor micro-batches 8/16
   and generation-memory reservations 0.60/0.70 in each of the twelve original
   standard-GRPO model–workload cases. Run each four-condition block on the same
   two physical GPUs, in a predeclared balanced order, with three new seeds.
   There are 36 allocations and 144 planned invocations. The ceiling is 72
   reserved GPU-hours. Full instrumentation is fixed; this does not isolate
   logging effects or hold generated responses identical.
2. **Prospective larger-model completion replication.** Test Qwen2.5-7B and
   Mistral-7B-Instruct-v0.3 on GSM8K, MATH, and the existing heavy-tail code
   workload. Cross the original compound levels c1/c2/c3 with two/four task
   GPUs and three new seeds. There are 36 configurations, 54 matched
   allocations, and 108 planned invocations. The ceiling is 198 reserved
   GPU-hours. The original fitted predictors are reused without refitting;
   forecasts are frozen before target execution. Level c1 includes parameter
   offload, so these levels are not a pure micro-batch/reservation experiment.

Both panels use A100-40GB GPUs and the existing VERL/vLLM stack. They broaden
the evidence within that setting, not across hardware or execution systems.
Seeds measure repeatability; they do not turn related configurations into
independent population samples.

## Execution gates

- Review the new protocols, source binding, device mapping, and CPU-only
  reconstruction before freezing execution inputs.
- Keep the Qwen-7B/GSM8K/c2/four-GPU diagnostics with seed 191 separate.
  Check that the actual trainer's completion metric is captured after its
  normal console print, and inspect retained worker logs. The first diagnostic
  validated the metric and worker operations, but exposed an incorrect
  worker-log retention path. A second, separately identified diagnostic checks
  that repair against the actual worker stdout. Each allocation is capped at
  30 minutes/four GPUs; neither is an evaluation observation.
- Preserve the initial prediction freeze and matrix byte-for-byte. The
  retention repair occurs after training has exited and changes neither
  settings nor predictions. Record a superseding execution-source freeze
  before any evaluation allocation, retaining the unused initial freezes.
- Preserve the historical panels exactly. A zero exit, returned operation,
  or logging repair is not a retrospective completed-step observation.
- Use immutable execution source, pinned dependencies/models/data, fresh
  processes, and pre/post device-idle checks.
- Freeze all target slots and predictions before starting a panel. Retain
  missing, invalid, failed, and unlaunched slots. Do not retry or replace
  evaluation slots to obtain a complete or more favorable panel.
- Record task GPU time and whole-allocation cost separately. Missing cost
  evidence is unknown, not zero. Account for diagnostics separately.

## Analysis and paper integration

- Reconstruct outcomes from raw evidence in an extracted, relocated copy
  without opening the original live data/model paths.
- Report useful approvals, failure approvals, margin errors, unresolved
  configurations, and costs by model. Do not select only configurations
  where a preferred method wins. Separate offloaded c1 from resident-actor
  c2/c3, and flag the ten targets whose fixed donor rule also changes GPU
  count. Input-only CPU prompt checks do not enter prediction.
- Report allocation-matched factorial contrasts by case, including
  incomplete-block coverage. Do not attribute compound-setting changes to
  one control.
- Edit the submission TeX directly. Keep detailed formulas and historical
  execution inventories in the supplement, and explain the practical
  question before each new result.
- Rebuild and inspect both PDFs and the extracted source bundle. Obtain a
  critical review of the actual revised manuscript and results.
- Create a new immutable release if publication is performed; do not
  overwrite the previously verified September 15 release.

The two study ceilings sum to 270 reserved GPU-hours; the two separate
diagnostics add at most four. These are upper bounds, not measured expenditure. Unresolved
scientific questions and author-controlled submission declarations remain
explicit even if the technical checks pass.
