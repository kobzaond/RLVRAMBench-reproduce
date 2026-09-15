# Larger-model test with expanded host resources

This separate follow-up asks whether the original 7B predictions remain
correct under an expanded host-resource condition intended to avoid the
host-memory exhaustion observed in the original test.
It is part of a focused measurement study, not a cross-platform benchmark.

The original panel could not label any four-GPU c2 configuration: seven
fresh attempts exhausted the 192 GiB host limit, and two earlier attempts
had setup failures. Those original records and unresolved labels remain
unchanged in `benchmark/estimation/results/`.

## Fixed comparison

The follow-up specifies all three original workloads at c2, with two or four
task GPUs and three new seeds. Its six configurations use eighteen process
slots in nine matched allocations. Each allocation reserves four A100 GPUs,
64 scheduler CPU units and 384 GiB of host memory. The two task-device counts
run in randomized order within that allocation.

The model, data, one-step schedule, batching, precision, memory controls and
measurement settings are inherited. This is not a controlled comparison of
host-memory capacity against the original panel: the processor request also
changes. There are no retries, replacement labels or further memory increases.

## Predictions and scores

`predictions.json` copies every original c2 prediction field, changing only
the target identifier. `protocol.json` maps each new identifier to its
original prediction. No fit, donor selection or threshold is updated.
The five distinct donors selected for six targets are existing measurements,
not new source acquisitions.

An approval is a predicted `within_margin` label. Read useful approvals,
memory-failure approvals and above-margin approvals together. Every repeated
label requires three eligible seeds; otherwise the target stays unresolved.
Always report unscored approvals and resolved coverage beside any score.
Neither a host kill nor a partial device peak establishes GPU feasibility.

## Observed outcomes

All eighteen invocations launched once. All nine two-GPU invocations have
validated generation-startup memory-admission failures; their three repeated
configuration labels are `memory_failure`.

All nine four-GPU invocations exit with code zero and record returning
actor-update and weight-synchronization calls, but lack the required final
training-step record. They remain unresolved under the unchanged completion
rule. This is missing completion evidence, not proof that no training occurred
and not a validated GPU-memory outcome. No scheduler host OOM was recorded.

Only three of six configurations are therefore scored. Donor copy
incorrectly approves all three and has three further unscored approvals.
Regression rejects the resolved failures but also has three unscored
approvals. The classifier approves nothing. An always-failure rule matches
the fitted methods on the failure-only resolved subset. There is no
within-margin target from which to estimate recall and no validated
completed peak for numerical error scoring.

All nine scheduler allocations are accounted for: 5.79 reserved GPU-hours,
including unresolved work. Task-provisioned invocation time is 4.89
GPU-hours; these are different, nonadditive scopes. The separate device
preflight used 0.08 GPU-hours.

Numerical peak errors use only targets with three completed invocations.
The donor-copy peak is the donor's measured peak, whereas regression supplies
a fitted estimate. Failed-run peaks are not imputed as completed outcomes.
Paired numerical differences require both invocations to complete.

## Files

- `protocol.json`, `matrix.csv`, `targets.csv`: fixed settings and inventory.
- `predictions.json`, `study_freeze.json`: inherited predictions and the
  separate pre-execution design seal.
- `execution_commits.json`, `execution_history.json`: immutable execution
  source, preflight and scheduler history.
- `results/configurations_flat.csv`: readable outcomes beside predictions,
  with seed coverage and unknown outcomes explicit.
- `results/processes.csv`, `results/pairs.csv`: every planned slot and pair.
- `results/evaluation.json`, `results/costs.json`: separate scores and cost
  views, not pooled with the original experiment.

The result files are populated only by independent raw-evidence
reconstruction. The browsing table contains outcomes and must not be used
as an input to blind prediction. Source acquisition, target invocation time,
whole scheduler allocation time and diagnostic preflight costs remain
separate.
