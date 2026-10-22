# Independent-factor experiment following the third supervisor review

The released larger-model panel confounds actor-parameter offload with its
compound configuration level. A post-hoc rule approving only four-GPU c1
settings recovers the classifier's six useful approvals while avoiding its
additional approvals. This does not constitute prospective validation of
that rule, but it motivates a new independent-factor experiment.

## Frozen questions and design

Cross actor-parameter offload on/off, per-GPU actor micro-batch 4/8, and
generation-memory reservation 0.45/0.70. Use four A100-40GB GPUs throughout,
Qwen2.5-7B and Mistral-7B-Instruct-v0.3, and the existing GSM8K, MATH, and
longer-prompt code workloads. Each model–workload–seed block executes all
eight conditions on the same physical devices in a randomized order.
Seeds 241–243 have no role in previous planned matrices.

This yields 48 configurations and 144 planned invocations in 18 allocations.
There are six exact original-c1 anchors and 42 independently crossed settings.
The old c2 and c3 settings do not occur: their reservations/batches differ.
The experiment is an outcome-motivated intervention in familiar cases,
not an unseen-family or cross-hardware validation.

The first candidate grid used reservation 0.45/0.60. Its CPU-only forecasts
had only three novel classifier/combined-rule disagreements, all in Mistral.
Critical design review recommended 0.45/0.70, which supplies seven such
disagreements across both families without expanding the run count.
No new GPU observation preceded this choice. The higher reservation tests
a failure-prone region, but old batch-16 failures do not establish the new
batch-4/8 outcomes. The design does not promise any mixture of labels.

Freeze the original ninety-configuration logistic and component-regression
fits without refitting. Also freeze four transparent admission rules:
offload-only; exact original-c1 only; low-reservation only; and
offload-or-low-reservation. No new donor-matching heuristic is introduced.
The exact-c1 rule is an anchor-only negative control, not a competitive
predictor on the crossed settings.

The classifier approves 29 settings, a subset of the combined rule's 36.
All seven differences involve offloaded, higher-reservation settings.
Six are Qwen settings and one is Mistral on GSM8K; these are seven
configuration comparisons, not 21 independent seed comparisons.
The primary prediction question is therefore **erroneous approvals avoided
versus usable settings lost**, not greater recovery by the classifier.
Report full-panel and crossed-setting scores, both families, all offload
groups, and the seven disagreements. Unresolved labels and partial evidence
of failure or margin exceedance remain visible.

Avoiding erroneous approvals without losing useful settings supports an
advantage over this specific rule on this selected panel only if no
still-unknown disagreement could reverse the zero-loss conclusion. Avoiding
errors while losing useful settings establishes a tradeoff, not unconditional
superiority. Partial validated failure or exceedance can establish known
unsuitability without resolving a three-seed label; report this separately.

The primary mechanism question compares offload on/off at each fixed
model, workload, seed, batch, and reservation. Report failure-inclusive
outcome transitions. Continuous memory contrasts require paired completion;
missing stages exclude their own contrasts, not an otherwise validated
completed run. Report both eligible and planned denominators.
The within-margin indicator measures completion and margin compliance, not
completion alone. Pooled continuous summaries average observed matched
completions, not the entire grid: a failing stratum can disappear. Keep the
fixed-batch/reservation comparisons primary and empty strata not estimable.

## Execution and interpretation

The randomized schedule uses two full eight-treatment Williams sequence
sets and two distinct additional sequences. Each condition appears two
or three times in each position; each ordered adjacent pair appears two
or three times. Each case receives three distinct sequences. This is not
exact within-case order balance. Actual order and attrition are retained.

Each allocation reserves four GPUs, 64 CPU cores, and 384 GiB host memory
for at most three hours. Every invocation has the same 900-second timeout
including setup and finalization, plus cleanup/idle checks. At most three
allocations run concurrently. The ceiling is 216 reserved GPU-hours,
not an expenditure forecast. Whole-allocation cost is reported separately
from overlapping invocation cost.

Use fresh processes, Ray namespaces, scratch/compiler caches, immutable
source and model/data snapshots, durable trainer completion records,
physical-device identities, allocation-wide process tracing, and pre/post
idle checks. Never retry, requeue, replace, extend selectively, reinterpret
timeouts as out-of-memory failures, or edit historical results.

Freeze every input, prediction, source closure, and output contract before
the first allocation. Inspect the first scheduled block as an execution
gate before launching the remaining blocks; its observations remain part
of the frozen panel, regardless of outcome. Release of the remaining
blocks depends only on implementation/evidence/cleanup validity, never on
which method wins or which memory label occurs. Any necessary execution
amendment must preserve the initial attempt and be independently sealed.

Repeated seeds assess repeatability in these selected cases. Show case
effects and observed within-case ranges. Do not treat 48 related grid cells
or 144 invocations as independent population samples. Generated responses
may differ between conditions, so comparisons concern total execution
effects rather than identical saved tensors.

Ties, simple-rule superiority, lost useful settings, failure-dominated
outcomes, and unresolved observations are legitimate findings. No result
is required for publication or for an acceptance-probability claim.

## Direct-TeX revision and release

Edit the authoritative IEEE Access TeX and its supplement directly.
Explain the shortcut limitation and the intervention before reporting
results. Consolidate the earlier larger-model execution history, avoid
another standalone supplementary section, and shorten repeated protocol
and limitation passages. Keep the scientific contribution a scoped
failure-aware measurement study.

Reconstruct new results from relocated raw evidence and exact source
snapshots; verify old results remain unchanged. Rebuild and visually check
the main PDF, supplement, and extracted source ZIP. Obtain critical review
of the actual revised manuscript, not a predicted acceptance probability.
Prepare a new versioned artifact without overwriting older releases.
