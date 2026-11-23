# Pre-execution amendments

Amendments A1–A6 below were made on 10 November 2026, before any E1–E3
invocation ran. They affect only smoke tests S1/S2, which are development
runs excluded from every estimate (PLAN.md §10). No matrix, seed, order,
setting, estimand, or analysis rule changed. A7 is the first amendment made
while a study (E3) was running; it follows PLAN.md §9.

## A1. Bundle verification on Karolina (setup S0)

`git bundle verify` in git 2.43.7 on Karolina refuses to run outside a
repository ("need a repository to verify a bundle"), which stopped
`setup_fta_storage.sh` after the copies, the container hash check, and the
runner environment. The remaining steps were completed by hand exactly as
scripted, with the bundle verified inside a temporary empty repository
(log: `/mnt/proj1/fta-26-70/rlvram/logs/setup-s0-continue.log`). The setup
script now does the same.

## A2. scikit-learn in the runner environment

Smoke jobs 5380614 (S1) and 5380616 (S2) failed at import, about 4 s after
start and before any GPU work: the runner imports
`memory_tuner.run_offload_factorial`, which imports
`memory_tuner.estimation_baselines`, which needs scikit-learn. The pin of
the repository (`publication-requirements.txt`: scikit-learn 1.3.0) was
installed into `/mnt/proj1/fta-26-70/rlvram/venv`
(log: `/mnt/proj1/fta-26-70/rlvram/logs/setup-amendments.log`). The setup
script now installs it.

## A3. Hydra output directory

Smoke jobs 5380690 (S1) and 5380691 (S2) failed in every invocation about
30 s after start: Hydra creates `outputs/<date>/<time>` in the working
directory, which is now the read-only source snapshot
(`PermissionError: [Errno 13] Permission denied: 'outputs'`). The earlier
launcher ran from a writable working copy and never hit this. The launcher
now passes `hydra.run.dir=/r/hydra hydra.output_subdir=null`, so Hydra
writes into the per-job scratch runtime. The schema check in
`ieee_resub/render.py` accepts these two Hydra keys explicitly. The dry run
was regenerated (121 invocations, 0 problems, 0 submitted) and the 36 tests
pass.

A new commit therefore replaces 0e83c81 as the source snapshot, and all five
protocols are frozen again against it; the protocols frozen against 0e83c81
are kept with the failed smoke jobs.

## A4. Draft protocols rebuilt for A3

`plan.py freeze-final` refuses to freeze when the source-file hashes differ
from the approved draft, so the drafts were rebuilt with `plan.py build`.
Compared with the approved drafts, only `created_at_utc` and the SHA-256
entries of `launcher/run_grpo_ieee_resub.slurm` and `ieee_resub/render.py`
changed; every matrix CSV is byte-identical.

## A5. Protocol location

The submit script accepts only the canonical protocol paths
`/mnt/proj1/fta-26-70/rlvram/protocols/<study>/<study>_protocol.json`.
The protocols frozen against 0e83c81, with the receipts of the failed smoke
jobs, were moved to `protocols/0e83c81/`, and the protocols were frozen
against 9f6faf9 at the canonical paths (hashes in `FROZEN_PROTOCOLS.txt`).
A freeze attempt against 711415f stopped at the source-hash check (A4) and
wrote no protocol. Smoke tests were resubmitted as jobs 5381160 (S1) and
5381161 (S2).

## A6. Smoke-test block claims

The runner claims each block directory permanently (no retry without an
amendment), so jobs 5381160 and 5381161 stopped at once because the failed
smoke attempts had already claimed `smoke1-s641` and `smoke2-s642`. Because
smoke tests are development runs, their earlier output was moved, intact, to
`output/failed-smoke-attempts/`, and S1/S2 were resubmitted with the same
protocols. The no-retry rule stays unchanged for E1–E3.

## A7. E3 runtime amendment 01: private /tmp, split configuration record, E1 levels

Recorded on 10 November 2026, while E3 (array 5381354) was running and before
any rerun, as `amendments/e3-amendment-01.json`, to be copied read-only to
`protocols/e3/amendment-01.json`. It is hash-chained to the frozen E3 protocol
(`f6ef9c52…`, unchanged and not refrozen) and records the complete source
hashes of the amended snapshot (equal to the rebuilt E3 draft's). Matrix,
seeds, settings, per-process timeouts, allocation shape, schedule, estimands
and PLAN.md are unchanged. Times are UTC.

*Defect 1 (runtime).* In job 5381357 (block
`e3-m7-code_heavy_tail-off1-r70-s631`, node acn10) the screen and the 100-step
run completed; the restore from step 50 wrote durable records 51–60 and then
failed in the weight sync of step 61 at 19:04:57 with `torch.AcceleratorError:
CUDA error: invalid argument` in `rebuild_ipc` (`training-5381357.log` lines
1317–1337). VERL names its colocated weight-transfer socket
`ipc:///tmp/rl-colocate-zmq-<Ray job id>-replica-<r>-rank-<k>.sock`, every
fresh Ray cluster has the same first job id, and the container saw the node's
shared `/tmp`. sacct shows job 5382462 (block
`e3-m7-code_heavy_tail-off1-r70-s632`) on acn10 since 18:26:24 on the other
four GPUs; its long run entered the weight sync of step 16 at 19:04:14, 43 s
before the error. Both have hung in `weight_sync` since, without any memory
diagnostic (all samples ≤ 30,756 MiB). A read-only look at acn10 showed one
shared socket, `/tmp/rl-colocate-zmq-01000000-replica-1-rank-0.sock`, created
0.4 s before the error in the host tmpfs `/tmp`, which is the containers'
`/tmp`; no runner, launcher or apptainer process had an inherited
`APPTAINER_`/`SINGULARITY_` bind variable. This is the defect of MI250X
runtime-amendment-01. The launcher now binds the invocation's own
`$SCRATCH_RUNTIME/tmp` (created and removed with the rest of the runtime) onto
`/tmp`, the only user bind of `/tmp`. Inside the container, before the trainer
starts, a per-invocation token written to that directory must be visible at
`/tmp`; otherwise the invocation stops with exit 97 (infrastructure,
unresolved). The dry run checks bind, token and wrapper for every invocation.
The only other container start, the environment capture, runs no Ray, vLLM
engine or socket and is unchanged. The launcher also accepts a restore source
below `preserved-checkpoints/` (and rejects `..`).

*Reruns.* Exactly the two slots of the colliding jobs, each submitted only
after its original attempt has ended unresolved at its timeout and its block
has finished: `e3-m7-code_heavy_tail-off1-r70-s631-restore` (12,600 s, about
22:11) and `e3-m7-code_heavy_tail-off1-r70-s632-long` (21,600 s, about 00:34
on 11 November). `runner.py rerun` claims `blocks/<block>-amendment-01`
permanently and uses the frozen matrix row (seed, settings, timeout) in a new
allocation of the E3 shape. The long run starts fresh. The restore resumes
from the copy of the step-50 checkpoint that the user preserved under
`/scratch/project/fta-26-70/rlvram/preserved-checkpoints/`. Before the claim,
at the start and after the invocation the runner requires this copy to equal,
file by file (SHA-256, size, mtime; 20 files), the manifest that the original
attempt recorded (`restore-source.json`, SHA-256 `ec9b49ee…`). The original
attempts and their receipts are not modified; the collector verifies this.
Neither rerun shares an allocation with the rest of its block (the restore's
source run, the long run's screen); those comparisons are reported with that
note.

*Defect 2 (assessment).* The long run of block `e3-q7-gsm8k-off1-r70-s633`
(job 5381356) completed 100 steps with all evidence, but Ray printed the first
configuration record as `{` followed by `'actor_rollout_ref': …`
(`training-5381356.log` lines 20–21), so `applied_actor_configuration` found no
start and the slot became unresolved (Slurm then marked task 1 FAILED).
`ieee_resub/assess.py` wraps the frozen parser: it is unchanged for every
console it accepts; only if it finds no start is a record that is exactly `{`
joined to the next record of the same process when that begins with
`'actor_rollout_ref':`, and the same parser runs again. Nothing is inferred.
The slot is reassessed from raw evidence, not rerun: `collect.py` takes the
raw status instead of the receipt only for the slot and job listed in the
amendment. On a read-only copy of all E1–E3 raw outputs, 99 of 100 training
logs parse identically with both parsers; the listed slot becomes within
margin (32,350 MiB, 100 durable steps, all scheduled validations and
checkpoints); no other slot of E1 (48 within), E2 (30 within, 10 above) or E3
changes.

*Defect 3 (analysis).* `analysis.py` maps the outcome instrumentation
`external` to the frozen estimand level `ext`; a test with collector-shaped
outcomes covers it. No execution is affected.

*Blocks 5–11* were never started or claimed (held in array 5381354). They are
cancelled there and submitted as array indices 5–11 (%3, same order) from the
amended snapshot with the amendment, so that they run with the private
`/tmp`; the runner accepts an amendment only for the blocks that it lists as
never launched. Tasks 3 and 4 continue under the original snapshot.

*Code.* `ieee_resub/amendment.py` (validation, `refresh`, `check`);
`runner.py` (`--amendment`, `rerun`); `collect.py` (rerun attempt as the
slot outcome, attempt ledger `attempts.json`/`.csv`, receipt/raw status
changes in the summary); `render.py` (amendment pass-through in the outer
scripts, `slurm/ieee_resub_e3_amendment.slurm`, rendered amendment commands);
`submit_ieee_resub.sh` (`--amendment`, `--indices`, `--rerun [--indices]`), which refuses
unless the read-only amendment matches its hash, the protocol hash and the
snapshot; `setup_fta_storage.sh --snapshot-only`. The drafts were rebuilt
with `plan.py build`: only `created_at_utc` and `source_files_sha256` changed;
every matrix CSV is byte-identical. The dry run was regenerated (121
invocations and 2 amendment reruns, 0 problems, 0 submitted).
