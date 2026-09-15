# Using RLVRAMBench

RLVRAMBench measures **whether a colocated language-model reinforcement
learning configuration completes within a specified device-memory margin**,
and whether that resource label transfers to another execution setting.
It is a measured systems dataset and a set of open transfer tasks, not a
language-model quality test, a trained predictor, or a live GPU service.

The released configurations run Group Relative Policy Optimization (GRPO)
with low-rank adaptation (LoRA). Generation, policy training, and weight
transfer share graphics processing units (GPUs). Memory demand is measured
externally for each device and linked to execution stages and failure logs.

## Prediction task

For each task, use its specified source measurements and target
configuration settings to predict:

1. `within_margin`: all three eligible repetitions complete with every
   sampled per-device peak at most 38,912 MiB.
2. `above_margin`: all three complete, but at least one exceeds that limit.
3. `memory_failure`: at least one eligible repetition has a diagnosed
   memory failure.

The limit leaves 2,048 MiB below each device's reported 40,960 MiB capacity.
It is a chosen operating margin, not an estimate of the physical failure
point. Eligibility and replacements concern the **resource endpoint**;
excluded attempts remain in separate completion and cost accounting.

| Track | Permitted source → target | Tasks | Queries |
|---|---|---:|---:|
| `model_transfer` | Another model, same workload and configuration level | 24 | 144 |
| `workload_transfer` | One workload's six-level profile → the other three workloads, same model | 12 | 216 |
| `gpu_count_transfer` | Matched two-GPU ↔ four-GPU settings at levels c2–c4 | 12 | 36 |
| `horizon_transfer` | Paired one-step → 100-step follow-up, same model/workload | 4 | 4 |

The 400 queries reuse 94 distinct target configurations and 98 distinct
source-or-target configurations. They are not 400 independent experiments.
The twelve single-process calibration settings and the other supporting
controls are not scored targets.

## Run the reference baseline

The small `benchmark/` directory is included with the reproduction code.
Python 3.11 or newer is sufficient for this interface; it has no third-party
dependencies and needs neither a GPU nor a virtual environment.

```bash
python3 benchmark.py verify --data benchmark
python3 benchmark.py baseline --data benchmark --output predictions.csv
python3 benchmark.py evaluate --data benchmark \
  --predictions predictions.csv --output scores.json
```

The reference rule copies the matched source label. Use `--track
model_transfer` (or another listed track) on both commands to score only
that track. `benchmark/baseline_predictions.csv` and
`benchmark/baseline_scores.json` contain the expected results.

## Evaluate your method

```bash
python3 benchmark.py inputs --data benchmark --track model_transfer \
  --output task_inputs.json
```

This writes separate task objects containing allowed source settings and
measurements, target settings, query IDs, and the source-target mapping.
The target objects contain **no target outcomes**. Make predictions from
each task independently and write:

```csv
query_id,predicted_state
<query_id from task_inputs.json>,within_margin
```

Replace the example with every query in the selected track. Then call
`evaluate` with the same `--track`. The evaluator checks exact query
coverage and outputs a confusion matrix, separate approval errors for
memory failure and margin violations, rejected within-margin settings,
explicit denominators, and deduplicated donor measurement costs.

Do not pool task objects to learn the target labels of one task from the
source labels of another. Do not use the target's outcomes, raw logs,
another target seed, or target-case supporting controls as extra input.
The public browsing tables join settings and outcomes for inspection,
not for blind prediction. Any extra measurements or use of these open
labels must be declared; this is **not a hidden leaderboard**. A method
developed on these outcomes requires new whole-case measurements for an
independent generalization claim.

## What this release can distinguish

Within a model, the four measured workloads have identical three-state
profiles on the six-level grid. Source-label lookup is therefore perfect
on workload transfer. That is a transparent reference check, not evidence
that the release discriminates sophisticated workload-aware predictors.

Cross-model and GPU-count transfers expose different approval errors;
paired horizon cases test a change in execution duration and its scheduled
operations. Repeated horizon labels agree in all four follow-up cases,
although two individual safe screens become above-margin long runs.
The three-seed rule already rejects that case. Stage-level and
state-release controls support interpretation of these bounded results.

The benchmark does not measure throughput optimality, output correctness,
training convergence, or unseen-model/general-hardware performance.
The c0–c5 indices are not a ranking of configuration usefulness.

## Inspect or reconstruct the evidence

See `DATA_DICTIONARY.md` for all fields, units, nulls, exclusions, and joins.
The Hugging Face default view displays actual configuration settings beside
their repeated outcomes; other views expose all configurations, eligible
runs, attempts, stages, and queries.

`export_benchmark.py --root EXTRACTED_ARTIFACT --output CURATED_DIRECTORY`
rebuilds these views from the validated result tables and original attempt
logs. It does not rerun training or recalculate the underlying scientific
measurements. For raw-evidence reconstruction, use the separate
`reproduce.py` procedure in the reproduction repository. Its output tables
can then be curated again and compared byte-for-byte.

The archived GPU launchers require the recorded software environment,
upstream models, a suitable allocation, and adaptation of cluster-specific
paths and scheduler settings. They are an advanced rerun workflow, not a
portable one-command interface.
