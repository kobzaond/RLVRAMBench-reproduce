# Memory-estimation comparison

This is a configuration-level evaluation within the RLVRAMBench measurement
study, separate from the original task-local transfer suite. It asks whether
model structure and configured work improve decisions beyond copying an
earlier measurement. It does not measure language-model quality or claim a
general memory estimator for arbitrary hardware and frameworks.

## What is predicted?

Each method predicts one label for a complete configuration, combining three
eligible seed outcomes:

- `within_margin`: all three runs complete and every sampled per-device peak
  is at most 38,912 MiB.
- `above_margin`: all three complete, but at least one peak exceeds that limit.
- `memory_failure`: at least one of the three eligible outcomes is a diagnosed
  memory failure.

An incomplete set of eligible seeds is unresolved, not a fourth prediction
class. Known failure evidence remains visible even when the repeated label
cannot be completed.

## Methods and separation

The historical fitting pool has 90 configurations: 72 boundary settings and
18 four-GPU settings. The three retrospective tests each withhold every
configuration from one model family, giving 60 fitting and 30 test
configurations. Seed processes are not separate fitting examples.
Historical labels were public during method design; this is not blind
development on an unseen dataset.

Three fixed references use the same permitted pool:

1. **Donor copy** selects a historical setting by workload, compound level,
   GPU count, model family and parameter count, without using target outcomes.
2. **Component regression with startup guard** fits a nonnegative weighted
   sum of architecture/configuration volume proxies to completed peaks.
   A separate one-sided resident-actor budget check can override its class.
   `peak_only_predicted_state` and `startup_guard_changed_state` distinguish
   the fitted estimate from this check. This is not an implementation of
   DNNMem, LLMem, or ReaL.
3. **Logistic classification** fits regularized class scores using all source
   labels. Its scores are not calibrated failure probabilities.

No failure peak is imputed as zero, capacity, or a survivor's completed peak.
Regression coefficients do not identify causal physical components.

The separate prospective panel specifies twelve Qwen2.5-7B configurations
and 36 fresh invocations in eighteen matched two-/four-GPU pairs. All
predictions were sealed before study execution. The 7B model belongs to a
known family, and older exploratory 7B workflows existed; those records are
excluded. This is not an unseen-family or cross-hardware test.

## How to read the result

Report useful within-margin approvals together with approvals of diagnosed
memory failures and above-margin completions. Exact three-state accuracy is
additional information, not a sufficient admission ranking. Evaluate each
held-out family separately; do not rank methods by a single pooled score
that hides incompatible error patterns.

`retrospective.json` contains every fold, fitted parameter, prediction,
confusion matrix and cost. The references do not uniformly improve on one
another: the regression recognizes Qwen labels well but approves four
memory-failing Granite configurations; the classifier rejects six usable
Qwen settings.

## Files and reconstruction

| File | Purpose |
|---|---|
| `protocol.json`, `amendment-01.json` | Original protocol and explicit prelaunch corrections |
| `model_metadata.json` | Header-derived counts, architecture, and source revisions; no model weights |
| `matrix.csv`, `targets.csv` | Frozen execution schedule and prediction settings |
| `retrospective.json` | Three leave-family-out comparisons |
| `fitted_model.json` | Final fit on all 90 historical configurations |
| `predictions.json` | Three frozen predictions per prospective configuration |
| `prediction_freeze.json` | Pre-execution timestamp and input/output hashes |
| `execution_commits.json`, `execution_history.json` | Exact execution sources and disclosed runtime repairs |
| `results/configurations_flat.csv` | One readable row per planned target, including coverage and all predictions |
| `results/evaluation.json` | Resolved-subset scores, full-panel coverage and approvals without resolved outcomes |
| `results/processes.csv`, `results/pairs.csv` | Every planned seed slot and matched pair, including unresolved evidence |
| `results/costs.json` | Separate task and allocation-time accounting, with missing records explicit |

The result files are produced only after raw-evidence reconstruction.
The flat table deliberately includes outcomes for inspection; use the frozen
target settings and permitted historical measurements when making predictions.
An unresolved target is not a correct prediction or a safe approval.
Always report the number of unscored approvals beside scores on the resolved
subset. Host-memory failures and software failures do not establish
GPU-memory outcomes.

The original larger-model panel is complete: 34 of 36 planned invocations
launched, 23 provide diagnosed startup memory-admission failures, and none
provides a completed training step. Thirteen slots remain unresolved,
including seven host-memory failures. Six of twelve repeated configuration
labels resolve, all to `memory_failure`. Donor copy incorrectly approves
all six. Both fitted references reject them, but so does an always-failure
rule: this failure-only subset cannot establish useful-configuration
discovery. The regression's three approvals are all unscored four-GPU
targets, not successful predictions.

The separately frozen [expanded-host follow-up](../host_capacity/README.md)
uses new seed slots and the same predictions. It does not replace these
outcomes or merge with their denominators.

From the reproduction code checkout, using the project-local publication
environment:

```bash
.venv-reproduce/bin/python -m memory_tuner.verify_estimation \
  --root . --refit --output estimator-verification.json
```

The output file must not already exist. This checks frozen inputs, recomputes
predictions and retrospective scores, and compares fresh fits without reading
prospective outcomes or overwriting the seal. Numerical tolerance is reported;
class labels must agree exactly. The archived backend-discovery warning limits
claims about CPU timing/thread control, not the recorded predictions.

The full historical acquisition comprises 308 attempts for 270 eligible
processes. Prospective donor copying selects ten distinct configurations,
using 34 historical attempts for thirty eligible processes. These are
deduplicated evidence costs, not demonstrated GPU-time savings.
