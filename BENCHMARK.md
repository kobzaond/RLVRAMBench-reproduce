# Using RLVRAMBench

RLVRAMBench measures **whether a colocated language-model reinforcement
learning configuration completes within a specified device-memory margin**,
and whether that resource label transfers to another execution setting.
It is a measured systems dataset and a set of open evaluation tasks, not a
language-model quality test or a general memory-prediction service.

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

For transparent comparison, `--rule always_approve` predicts within-margin
completion for every target, while `--rule always_reject` predicts memory
failure for every target. The latter is a deliberately crude diagnostic,
not evidence that every rejected setting actually fails.
`benchmark/reference_scores.json` records all three reference rules.

## What does the score mean?

The primary question is whether a method identifies usable configurations
without approving unsuitable ones. The evaluator therefore reports:

- `within_margin_recall`: usable targets approved / all usable targets.
- `within_margin_approval_precision`: usable targets approved / all approvals.
- Separate approvals of memory-failing and above-margin targets, with counts
  and denominators.
- `three_state_accuracy`, for exact prediction of all three outcome categories.

An empty denominator is `null`, not zero error or perfect performance.
Always rejecting everything gives no approval errors but recovers no usable
targets. Always approving everything recovers every usable target while also
approving every failure. Neither is a successful resource-selection method.

The source-copy baseline has 88% pooled exact accuracy, but more than half
the queries are the already-solved workload transfer. On GPU-count transfer,
copying labels scores 55.6% accuracy and approves two memory-failing targets;
always approving scores 72.2% and approves four. Higher accuracy can therefore
give worse admission decisions. Report each track and its error types,
not the pooled accuracy as a standalone benchmark ranking.

The static evaluator's donor-cost fields describe the evidence supplied by
the task. They do not meter a method's actual information use, adaptive
requests, or GPU-hours saved. These counts are consequently the same for
constant and source-copy rules supplied with the same task.

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

## Separate model-family estimation comparison

`benchmark/estimation/` uses a different information contract from the
task-local suite above. Its ninety historical configurations supply three
leave-family-out tests: fit on sixty configurations and predict the thirty
from the remaining family. The references are donor copying, empirical
component regression with a startup guard, and regularized logistic
classification. All seeds and GPU-count variants of the test family remain
outside fitting. Historical labels were visible during method design.

The [estimator guide](benchmark/estimation/README.md) gives the frozen
specification, prediction files, per-family scores and a CPU verification
command. The references have different failure and rejection patterns; none
uniformly improves on the others. Do not mix these fitting pools with
the original task-local donor rules or pool their repeated targets into
an inflated test-set size.

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

## Budgeted decisions with independent evaluation runs

The separate `benchmark/decision/` protocol asks a more practical question:
**how many usable settings can existing donor measurements identify before
new target screens are available, and what approval errors remain?**

A prospectively specified panel changes the workload shape and scheduled
operations in four known model/task combinations. Each of twelve candidate
configurations has one designated five-step screen and three different-seed
evaluation runs. The screen is never part of its own evaluation label.
The frozen protocol, matrix, prelaunch amendment, and collection records
preserve that separation.

`decision_benchmark.py` compares source-label copying, a fixed additional
donor-headroom guard, and direct screening without donors. They all query
the same preassigned candidate order, under zero through three recorded
target attempts per case. A charged screen replaces the initial decision;
an unresolved screen causes abstention. Evaluation runs remain hidden from
the policy even though their records are public for inspection.

```bash
python3 decision_benchmark.py inputs \
  --protocol benchmark/decision/protocol.json \
  --rule source_copy --output decision_inputs.json
python3 decision_benchmark.py replay \
  --protocol benchmark/decision/protocol.json \
  --attempts benchmark/decision/results/attempts.json \
  --output decision_scores.json
```

Interpret within-margin recall jointly with both approval-error counts and
the actual spent budget. The output preserves per-case decisions and every
charged observation. Unresolved evaluation labels remain explicit; the
sensitivity ranges exclude states contradicted by known failure or margin
evidence. Overlapping known-evidence flags must not be added as disjoint
outcome categories.

The cost ledgers distinguish:

1. Requested target attempts, including failed setup and any permitted
   replacement: incremental effort when a donor profile already exists.
2. Donor acquisition: nine historical recorded attempts per case for the
   specified transfer rules; zero for direct screening.
3. Hidden evaluation collection: research cost, never policy input.

Cold-start accounting adds donor acquisition. It does not compare against
every possible direct-testing strategy with that larger budget. Attempts
are not equal amounts of computation, and this is offline replay of frozen
rules, not an online deployment trial or a claimed throughput optimizer.
The four cases remain four cases regardless of the number of seeds,
budgets, or methods displayed.

### Results of the frozen admission panel

All 48 slots have validated outcomes, without retries or unresolved
attempts: 24 processes complete within margin, eight complete above it,
and sixteen fail for memory. The twelve candidate evaluation labels
are six within margin, two above margin, and four memory failures.

With source measurements already available, source copying recovers all
six usable candidates with no new test. The guard rejects the two usable
Qwen c3 candidates, giving 75% case-averaged recall before testing and
100% after one target attempt per case. Those two candidates are queried
first, so copy and guard decisions are structurally identical at every
positive budget. Their agreement is not independent method evidence.

Direct screening reaches case-averaged recall of 0%, 25%, 75%, and 100%
at budgets of zero through three attempts per case. It makes no approval
error in this panel. Nor do the transfer rules. The fixed query order
determines when usable settings are discovered; no better-order claim
follows from these results.

At the first tested common per-case budget attaining full recovery,
the incremental target-attempt totals are zero, four, and twelve for
copy, guard, and direct screening. These are not optimal-acquisition
lower bounds. Adding prior donor
investment gives 36, 40, and 12 attributed attempts, respectively.
The former comparison is useful for reuse of an existing profile; the
latter prevents that profile being presented as free in a cold start.
Neither is a GPU-time saving estimate.

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
