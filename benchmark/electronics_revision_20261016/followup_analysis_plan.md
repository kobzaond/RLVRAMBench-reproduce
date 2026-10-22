This plan is fixed in the execution source commit before any of the
reservation, release, or timing panels is submitted.

All three panels use the documented VERL/vLLM container, SHA-256
`3dc581ca780cd19d565445b8a02e5daa98b350b3d2116d4d64476a64a2614912`.
Each matrix and model/data snapshot is separately hashed. Old outcomes
motivated the selected cases; these are fresh within-case comparisons,
not blind sampling from deployments. No automatic retries, requeues,
replacement seeds, or outcome-dependent grid extensions are allowed.
Never substitute a failed partial peak for a completed-run peak.
All planned slots remain in denominators, with unresolved and unstarted
work separate from the three scientific memory outcomes.

**Reservation: 144 invocations, 48 configurations.** Qwen7B and Mistral7B,
GSM8K and longer-prompt code, actor offload on/off, reservations
0.45/0.50/0.55/0.60/0.65/0.70, and three fresh seeds. Actor micro-batch 8,
four task GPUs, one training step, full allocator instrumentation, and
generation-state release are fixed. Each of twelve same-device blocks
executes all twelve conditions. The full twelve-sequence Williams design
balances condition positions and directed adjacency across the selected
blocks; it does not eliminate host contention or thermal carryover.

Primary reporting: per-case outcome curves with all seeds, repeated labels
where all three seeds are eligible, and tested transition intervals. Do not
assume monotonicity or interpolate an untested label. Compare offload within
each reservation and reservation within each offload state. Peak differences
require joint completion and must show that denominator. Endpoints are
repeated contemporaneously; historical endpoints are supporting context.

Prediction reporting reuses the original frozen logistic and component
regression coefficients without refitting, and the original admission rules.
“Lower reservation” in those rules still means exactly 0.45; it is not
retuned after observing intermediate values. Report the four intermediate
fractions separately from the familiar endpoints. For each model, workload,
and offload state, report within-margin recall and memory-failure and margin
approvals. Numerical errors use the maximum completed peak of three eligible,
completed repeats only. Failed configurations have no numerical target.
LLMem's native OPT/Gemini comparator has no validated mapping to these
LoRA/FSDP/vLLM targets, so its entries explicitly abstain as unsupported.
Its native numerical validation is reported separately, without counting
abstentions as correct rejections or making an end-to-end accuracy claim.

**State release: 36 invocations, 12 configurations.** Two existing GSM8K
anchors are reproduced with their original two task GPUs and workload:
Qwen1.5B at reservation 0.70 (both modes previously completed), and Qwen3B
at 0.65 (release previously changed completion). Within each anchor, compare
release/resident at original reservation minus 0.10, the original fraction,
and plus 0.05, for three fresh seeds. Six conditions run on the same physical
task GPUs within each block in recorded random order. These blocks are
randomized, not claimed perfectly balanced. The unchanged runner reserves
four GPUs but assigns only two to the task; report allocation and task GPU
cost separately and do not add them together.

Primary reporting: release-related completion changes at each tested
reservation, completed-pair peak contrasts, and diagnosed failure contexts.
The two base workloads remain distinct. This panel tests sensitivity to
reservation within each base, whereas the original six-row table did not.

**Iteration cost: 24 invocations.** Qwen7B/GSM8K and Mistral7B/longer-prompt
code each receive three same-device four-condition blocks. The common anchor
offloads actor parameters, reserves 0.45, and releases generation state.
Its three contrasts respectively disable actor offload, raise reservation
to 0.70, or retain generation state. All request twenty steps with actor
micro-batch 8, the same global batch/rollout count/length caps/seed, no
validation or checkpoint writing, and external-only GPU instrumentation.

The first two updates are reported separately as warm-up. The primary
timing window is updates 3–20, and only complete common work contributes to
the paired timing comparison. Report summed trainer step duration, generation,
reference-scoring, actor-update and synchronization intervals, generated
response tokens, prompt-plus-response tokens, and sequences processed.
Normalize token rates using observed work, and retain differing generated
lengths rather than asserting tensor-identical work. Host-memory sampling
uses the Slurm batch cgroup at one-second intervals, recording anonymous
memory and charged file cache separately; missing cgroup access is missing
secondary evidence, not zero host memory or a GPU failure.

Report end-to-end invocation time and task GPU-hours separately from the
warm window and outer allocation GPU-hours. A failed configuration is not
a faster alternative. Timed stage intervals do not isolate communication
kernels. Twenty steps do not establish convergence, final quality,
checkpoint feasibility, or throughput optimality. The optional paired
100-step larger-model extension is outside these three frozen panels.
