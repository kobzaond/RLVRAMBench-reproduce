# Benchmark-specific VERL snapshot

This directory is the benchmark's instrumented and compatibility-patched
VERL source snapshot, not an unmodified upstream distribution. Original
copyright and attribution notices remain in the source files.

The accompanying `LICENSE` is a byte-preserved copy of the Apache-2.0
license supplied by the installed VERL distribution
(`verl-0.10.0.dev0.dist-info/licenses/LICENSE`). This preserves upstream
terms; it does not select a license for the new benchmark repository.

The benchmark adds phase markers, optional per-worker allocator hooks,
response-length tracing, and compatibility repairs. The September 2026
revision additionally changes native checkpoint configuration serialization;
its implementation and limits are documented in
`paper/scientific_revision_checkpoint_amendment_2026-09-13.md`.
These changes do not introduce a new GRPO objective or scheduling policy.

The November 2026 resubmission copy (`experiments/ieee_resubmission_2026-11/`)
adds three opt-in measurement controls, all disabled by default: fixed-length
generation (`rollout.fixed_response_length`), canonical batch ordering before
sequence balancing (`RLVRAM_CANONICAL_BATCH_ORDER`), and a host-side batch-shape
trace (`RLVRAM_SHAPE_TRACE`). Resumed runs additionally mark checkpoint loading
as its own phase. See `verl_patch/ieee_resub_verl.patch` there.
The exact source head and clean-worktree status for each experiment are in
the immutable environment and allocation records.

The pinned TransferQueue dependency is distributed separately in
`third_party/TransferQueue/`, including its own license. Upstream model and
dataset terms are recorded in `paper/PROVENANCE.md`; model weights and
optimizer checkpoints are not redistributed.
