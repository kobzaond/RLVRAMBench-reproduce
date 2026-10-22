This plan is fixed in the execution source commit before submission.

The 96 native-KL invocations cross two models (Qwen2.5 1.5B and 3B),
GSM8K and longer-prompt code, LoRA rank 64 versus full-parameter training,
actor parameter offload on/off, reservations 0.45/0.65, and three seeds.
Each model/workload/seed block executes eight fresh processes on the same
four GPUs in a recorded randomized order. Actor micro-batch is 4, optimizer
offload is disabled, and generation-state release is enabled. Each process
requests five training steps, without validation or checkpoint writing.
The 160-example preparation limit is fixed; realized filtering and generated
lengths remain part of the observed workload. These are selected cases,
not deployment-distribution samples.

The primary outcomes are diagnosed memory failure, completion within the
38,912 MiB margin, completion above it, and unresolved noncompletion.
Report all 108 planned slots, including anything unstarted. Never substitute
a failed run's partial peak for its hypothetical completed-run peak. No
automatic retry or post-outcome boundary refinement is authorized by this
protocol. Configuration summaries require all three seeds; otherwise retain
the missing evidence instead of assigning a complete repeated label.

Primary contrasts within native-KL blocks:

1. Full versus LoRA completion, keeping model, workload, seed, reservation,
   offload, requested work, and physical devices fixed.
2. Whole-run and actor-stage peak differences in jointly completed pairs,
   reported separately from completion changes and with their denominator.
3. Offload contrasts within each adaptation mode and reservation, with the
   same separation of categorical and continuous outcomes.

Report each model/workload and individual seed before any equal-case mean.
With three repeats and selected cases, no population accuracy or calibrated
failure-probability claims are intended. Preserve realized token/length
differences; five-step elapsed time includes startup and compilation and
does not establish steady-state throughput or convergence.

The native full-parameter implementation instantiates a separate frozen
reference role, whereas LoRA reuses its frozen base. Consequently, native
mode contrasts combine trainable-state, reference-layout, and synchronization
changes. Actual rank-zero parameter metadata must demonstrate all parameters
trainable without adapters in rank-zero runs; LoRA must train adapters only.

Twelve additional no-KL invocations cross LoRA/full and actor offload on/off
for Qwen3B/GSM8K at reservation 0.65 and three fresh seeds. The reference is
absent in both modes. These four-condition same-device blocks test the
adaptation contrast without a reference role. Disabling KL changes the
objective: do not pool these outcomes with native-KL runs, claim a pure
reference-removal effect across the separately allocated studies, or infer
equivalent model quality.

Three earlier two-step runs are development-only compatibility checks.
They cannot supply a missing confirmatory seed or enter the above contrasts.
Their observations motivated feasibility of this execution mode, not a
blind claim about its outcomes.

Each allocation is capped at three hours, each payload at 900 seconds, and
the array at four concurrent four-GPU allocations. Cleanup or physical-device
identity failure stops subsequent work in that block. GPU-time reporting
separates task execution from full allocation time.
