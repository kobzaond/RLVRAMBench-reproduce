# IEEE-resubmission GPU experiments E1–E3: frozen-protocol draft

Status: **draft for approval, 10 November 2026.** Nothing has been submitted,
nothing has been written to the cluster, and no git remote was touched. After
approval, steps S0–S5 (Section 11) turn this draft into frozen protocols.

Repository root R = `rl/`; this directory is
`R/experiments/ieee_resubmission_2026-11/`. Paper references are to
`R/paper/submission/ccpe_submission/RLVRAMBench_CCPE.tex` (CCPE version).

## 1. Purpose and the reviewer comments answered

| Exp. | Reviewer comments | Question |
|---|---|---|
| **E1** | R1.3, R3.3, R4.3 (IEEE Access, Sep 2026) | Does the actor-stage micro-batch contrast (3,905 MiB, equal-case mean; whole run 7 MiB) persist when every condition processes *identical tensor shapes* (same prompts, same order, every response forced to the response cap)? Does the logging interaction persist with identical shapes? |
| **E2** | R3.7 | How often do four near-boundary configurations give each outcome over ten fresh one-step repeats (exact binomial intervals)? |
| **E3** | R3.2, R5.M6 | Do startup-feasible 7B settings stay within margin over 100 steps with validation and checkpoints; how much later growth; does a restore from a checkpoint change the peak; do one-step screens agree with the long runs? |

The CCPE text currently only states these limits (status table in
`ccpe_submission/IEEE_REVIEW_STATUS_AND_EXPERIMENT_PLAN.md`, items E1–E3).

## 2. Common definitions (identical to the paper, Sections 3–4)

- **Slot** = one planned run (configuration × seed); **attempt** = one actual execution; each slot gets at most one attempt unless a written amendment (Section 9) adds one.
- **Fresh process** for every invocation; **same-device block** = invocations of one block on the same physical GPUs of one Slurm allocation, idle gate (<1,024 MiB used, 0 % utilization, no processes, twice) before and after each period.
- **Device capacity** 40,960 MiB (A100-SXM4-40GB); **operating limit** 38,912 MiB (2,048 MiB margin).
- **Within-margin completion**: eligible completion and every 100 ms NVML sample on every task GPU ≤ 38,912 MiB; **above-margin**: eligible completion with a larger sample; **memory failure**: eligible non-completion with an explicit GPU-memory diagnostic (CUDA OOM allocation error, or vLLM startup rejection "free memory on device … desired GPU memory utilization"), classified by `memory_tuner/log_parser.classify_failure_detailed` and `electronics_failure_evidence.has_gpu_memory_evidence`. Timeouts, host OOM, Ray/port failures and missing evidence are **unresolved**, never memory failures.
- **Eligibility** (paper Section 3.7): trial record, environment record (clean frozen commit, container SHA-256 `3dc581ca…2912`, torch 2.10.0+cu129, vLLM 0.18.0, model revision, dataset hashes), execution log, stage trace, per-GPU NVML trace, physical-device identity (probe/exec/worker records). **Completion** additionally needs exit 0, durable trainer step records for *every* requested step (`trainer-metrics/step-N.json`, written after the normal console print), the parameter inventory, and every scheduled validation and checkpoint observed in the stage trace (Section 7.4). E1 additionally needs a valid batch-shape trace with every response length equal to the cap; otherwise the run is unresolved.
- **Three-state label** over a configuration's planned slots: unresolved if any slot lacks an eligible outcome; else memory failure if any; else above margin if any; else within margin.
- **Stage peak** = max NVML sample within a host-marked interval (markers do not synchronize GPUs and do not identify allocation ownership); **whole-run peak** = max over all samples and task GPUs (completed runs only). Stage contrasts use complete blocks (Section 7).
- **Full instrumentation** = NVML 100 ms + stage markers + allocator tracing with GPU synchronization before each traced worker operation (`ALLOCATOR_TRACE_ENABLED=True`, sync 1). **External-only** = NVML + stage markers (`False`, sync 0).

## 3. E1 — controlled-length micro-batch × logging contrast

### 3.1 Design
Same-device blocks of four conditions — actor micro-batch 8 vs 16 × external-only vs full instrumentation — each a fresh process on the same two GPUs (one 2-GPU allocation per block, as in the same-device replication). Reservation 0.60, generation-state release on, actor offload off, one training step, no validation, no checkpoint. 4 cases × 3 seeds = **12 blocks, 48 invocations** (`matrices/e1_matrix.csv`).

### 3.2 How fixed-length generation works in this VERL/vLLM 0.18 stack
- VERL's rollout config already has `ignore_eos` (`workers/config/rollout.py`), which `vLLMHttpServer.generate` passes into vLLM `SamplingParams`. `max_tokens` is pinned by `LLMServerClient` to `rollout.response_length` and is never reduced for these workloads because `max_model_len = prompt cap + response cap` and prompts are filtered to the prompt cap.
- New opt-in flag **`actor_rollout_ref.rollout.fixed_response_length`** (default False; RolloutConfig field + `rollout.yaml`). When true, the server sets `ignore_eos=True` **and** `min_tokens=max_tokens` for every request of that engine. `ignore_eos` alone already prevents EOS from ending a request; `min_tokens` additionally masks EOS/stop tokens until the cap so that no text follows a sampled EOS. With `ignore_eos`, vLLM's `update_from_generation_config` should also not add the generation-config EOS IDs as stop tokens; because the vLLM 0.18 source exists only inside the container (not inspected), smoke test S1 verifies empirically that every response reaches the cap. The flag also applies to validation requests; E1 has none.
- Same prompts in the same order: `data.shuffle=False` with `train_max_samples = train batch` takes the same first N rows for every condition and seed (VERL `RLHFDataset` with `shuffle=False` → `np.arange`).
- **Canonical batch order** (`RLVRAM_CANONICAL_BATCH_ORDER=1`, E1 only): the V1 replay buffer returns trajectories in asynchronous completion order and `get_seqlen_balanced_partitions` keeps indices in that order within each rank, so identical lengths could still form different micro-batches. Before sequence balancing the trainer sorts the sampled batch by (prompt token IDs, rollout index); balancing, dispatch to ranks (contiguous chunks) and micro-batching (contiguous, `use_dynamic_bsz=False`) are then deterministic. Grouping by UID (GRPO advantages) is unaffected. Micro-batch 16 batches are unions of consecutive micro-batch 8 batches.
- **Shape evidence** (`RLVRAM_SHAPE_TRACE`, host-side only): after balancing, the trainer appends one JSONL record per step: SHA-256 of the prompt token IDs in dispatch order and as a multiset, of the (prompt length, response length) vector in dispatch order and as a multiset, of response lengths, and (diagnostic, expected to differ) of response token IDs; the full per-sequence prompt/response length vectors, rollout indices, and per-rank actor and log-prob micro-batch token totals. **Identical shapes** within a block ⇔ all four runs share `dispatch_prompt_tokens_sha256` and `dispatch_length_pairs_sha256`, and every response length equals the cap.
- Code: `verl_patch/ieee_resub_verl.patch` (8 files, 388 lines) applied to a copy in `instrumented_python_packages/`; `R/instrumented_python_packages` is untouched (test `test_patch_integrity.py`).

### 3.3 Memory risk of forcing responses to the cap, and case selection
Natural realized lengths (`profiles/review_revision/realized-workload-trials.csv`) are far below the caps for GSM8K (e.g., Qwen-3B: 107 + 293 tokens vs cap 2,048). I fitted the actor-stage peak per model on micro-batch tokens (micro-batch × realized mean prompt+response length) over the 144 original factorial runs (`profiles/standard_grpo/factorial-trials.csv`; residual SD 268/673/496 MiB for Granite/Phi/Qwen-3B; slopes 0.39/1.16/0.79 MiB per token, which mirror vocabulary size) and predicted forced-length peaks (prompt mean + response cap). The generation (whole-run) peak at 0.60 does not change with response length (KV cache is pre-reserved); the reservation does not affect the actor stage because generation state is released then (allocator evidence: 9 MiB, paper line 1666), so changing the reservation cannot make a case safer.

| Case (reservation 0.60) | Forced tokens/seq | Predicted actor peak b8 / b16 (MiB) | Predicted actor contrast | Natural whole-run peak | Decision |
|---|---:|---:|---:|---:|---|
| Granite-3.3-2B / GSM8K | 2,194 | 17,527 / 24,443 | 6,915 | 32,218 | **keep** |
| Phi-4-mini / longer-prompt code | 1,688 | 30,471 / **46,081** | 15,611 | 35,866 | drop: b16 exceeds device capacity |
| Qwen2.5-3B / GSM8K | 2,155 | 25,757 / **39,445** | 13,689 | 33,036 | drop: b16 above the limit, OOM likely |
| Qwen2.5-3B / longer-prompt code | 1,744 | 23,146 / 34,224 | 11,078 | 33,316 | **keep** (closest; actor stage would set the whole-run peak, predicted whole-run contrast ≈ 900 MiB) |
| Phi-4-mini / MATH | 588 | 20,298 / 25,736 | 5,438 | 36,128 | **added** (only Phi workload predicted to fit) |
| Qwen2.5-3B / MATH | 617 | 15,987 / 19,906 | 3,919 | 33,458 | **added** (replaces Qwen-3B/GSM8K; keeps three models) |

The fit extrapolates (fitted micro-batch tokens ≤ ~17.6k; forced b16 up to 35k), so the Qwen-3B/longer-prompt-code b16 condition is the main residual risk (≈4.7 GB headroom to the limit). Smoke test S1 runs exactly that condition first (Section 10). Both replacements are among the twelve same-device-replication cases, so natural-length references exist. Predictions are frozen in `matrices/protocol_draft/e1.json` as descriptive expectations, never as outcome requirements.

### 3.4 Fixed settings (copied from `benchmark/replication_control/replication_control_matrix.csv`, condition `base`, reservation 0.60)

| Case | Prompt/resp./model len | Max seqs | Rollouts | Global batch | Train samples | Log-prob mb |
|---|---|---:|---:|---:|---:|---:|
| Granite/GSM8K | 1024/2048/3072 | 64 | 2 | 32 | 32 | 4 |
| Phi/MATH | 512/512/1024 | 128 | 2 | 64 | 64 | 8 |
| Qwen-3B/MATH | 512/512/1024 | 128 | 2 | 64 | 64 | 8 |
| Qwen-3B/longer-prompt code | 1536/1024/2560 | 64 | 2 | 32 | 32 | 4 |

All: LoRA rank 64, alpha 32, KL loss 0.001 (low_var_kl), lr 3e-6, gradient checkpointing, padding removal, optimizer offload off, parameter offload off, release on, TP 1, 2 task GPUs, 100 ms NVML, response-length trace on, `train_max_samples` = global batch, `shuffle=False`. Code reward `memory_tuner/code_shape_reward.py`. Models (`paper/PROVENANCE.md`): Granite `707f574c…`, Phi `cfbefacb…`, Qwen-3B `aa8e7253…`.

### 3.5 Seeds, order, placement, timeouts
Seeds **601, 602, 603** (shared by the four cases, as 211–213 in the replication). Order seed **2026111001**: a Williams 4×4 square (each condition once per position and each directed adjacency once per square); symbol→condition mapping `{0: b16-ext, 1: b08-full, 2: b16-full, 3: b08-ext}`; case i receives Williams rows {i, i+1, i+2} mod 4 assigned to its seeds at random (each row used three times; each condition occurs three times in each position and each directed adjacency three times overall; three distinct sequences per case); block (array) order shuffled with the same seed. Allocation: `--gpus=2 --cpus-per-task=32 --mem=192G --time=02:00:00`, array `0-11%3`. Per-process timeout 1,200 s (replication: 720 s).

| Array | Block | Order |
|---|---|---|
| 0 | e1-q3-math-s603 | b08-ext → b16-ext → b16-full → b08-full |
| 1 | e1-q3-code_heavy_tail-s603 | b16-full → b08-ext → b08-full → b16-ext |
| 2 | e1-q3-code_heavy_tail-s601 | b16-ext → b08-full → b08-ext → b16-full |
| 3 | e1-p4-math-s602 | b16-ext → b08-full → b08-ext → b16-full |
| 4 | e1-g2-gsm8k-s603 | b08-full → b16-full → b16-ext → b08-ext |
| 5 | e1-q3-math-s602 | b16-ext → b08-full → b08-ext → b16-full |
| 6 | e1-p4-math-s603 | b08-ext → b16-ext → b16-full → b08-full |
| 7 | e1-g2-gsm8k-s602 | b08-ext → b16-ext → b16-full → b08-full |
| 8 | e1-q3-code_heavy_tail-s602 | b08-full → b16-full → b16-ext → b08-ext |
| 9 | e1-q3-math-s601 | b16-full → b08-ext → b08-full → b16-ext |
| 10 | e1-g2-gsm8k-s601 | b16-full → b08-ext → b08-full → b16-ext |
| 11 | e1-p4-math-s601 | b08-full → b16-full → b16-ext → b08-ext |

Natural-generation reference: no new runs; the same-device replication at 0.60 (`profiles/electronics_revision/stage_paired_contrasts.csv`, full instrumentation) and the batch-by-logging control (`profiles/review_revision/control-trials.csv`).

## 4. E2 — ten fresh repeats near the boundary

Four configurations × seeds **611–620** = **40 one-step invocations** (`matrices/e2_matrix.csv`). Settings are copied field by field from the original rows (test `test_e2_templates_and_balance`):

| ID | Source row | Settings | Earlier outcomes |
|---|---|---|---|
| (a) `q3-gsm8k-rel-r65` | `benchmark/electronics_revision_20261016/release/matrix.csv` `release-q3-gsm8k-s501-release1-r65` | Qwen2.5-3B, GSM8K 256/1024/1280, 256 seqs, n=5, batch 128, mb 16, log-prob 8/8, 0.65, release on, offload off, 2 task GPUs | release experiment 3/3 within (36,544–36,750); release-sensitivity 2/3 within, 1 above (39,244) |
| (b) `g2-math-c3` | `memory_tuner/rlvram_strengthening_granite_confirmatory.csv` `…math_short-c3-s92` | Granite, MATH 512/512/1024, 128 seqs, n=2, batch 64, train 256, mb 16, 0.70, offload off, release on, 2 GPUs | 9/9 within (boundary + instrumentation control), max 38,718 (194 MiB headroom) |
| (c) `p4-gsm8k-c3` | `memory_tuner/rlvram_v2_cross_family_confirmatory.csv` `…gsm8k_long-c3-s42` | Phi-4-mini, GSM8K 1024/2048/3072, 64 seqs, n=2, batch 32, mb 16, 0.70, offload off, release on, 2 GPUs | 9/9 above (39,550–39,570) |
| (d) `q7-gsm8k-off0-r60` | `benchmark/electronics_revision_20261016/reservation/matrix.csv` `reservation-q7-gsm8k-s452-off0-r60` | Qwen2.5-7B (`a09a3545…`), GSM8K, mb 8, log-prob 4/4, 0.60, offload off, release on, 4 GPUs | 3/3 within at 38,560 (352 MiB headroom); 0.65 above |

**Instrumentation: full** for all four — the configuration of every original run (boundary runs used the launcher default `ALLOCATOR_TRACE_ENABLED=True`, sync 1, before it became explicit on 13 Sep 2026; the release and reservation panels set True/1). This deviates from the internal plan note ("external-only"), on purpose.

Placement: one 4-GPU allocation per seed (block), `--gpus=4 --cpus-per-task=64 --mem=384G --time=01:40:00`, array `0-9%3`, like the release-sensitivity panel (2 task GPUs inside a 4-GPU allocation, where (a)'s label changed); (a)–(c) use ordinals 0;1, (d) uses 0–3. Order seed **2026111002**: Williams rows used twice plus two random extra rows; each configuration occurs 2–3 times in each position; block order shuffled (table in `matrices/protocol_draft/e2.json`). Per-process timeout 900 s (as in both panels).

## 5. E3 — 100-step runs with validation, checkpoints, screens and restore

Configurations (iteration-cost panel workloads, `benchmark/electronics_revision_20261016/timing/matrix.csv`; offload/reservation from the reservation panel): Qwen2.5-7B/GSM8K and Mistral-7B-Instruct-v0.3 (`c170c708c41dac9275d15a8fff4eca08d52bab71`)/longer-prompt code; 4 GPUs, micro-batch 8, log-prob 4/4, release on, optimizer offload off: **offload on at 0.70**; **offload off at 0.60 (Qwen, 38,560 MiB in the reservation panel) / 0.55 (Mistral, ≤38,334 MiB)**. Seeds **631, 632, 633**.

Block = (configuration, seed), one 4-GPU allocation: period 1 **screen** (1 step), period 2 **long run** (100 steps), period 3 **restore** (seed 631 only). Fixed order (the screen precedes the run, as in the paired duration test; the restore follows its source). 12 blocks, **28 invocations** (12 screens, 12 long runs, 4 restores); block order shuffled with order seed **2026111003**. `--time=11:30:00`, array `0-11%3`.

Schedule (paired duration test conventions, `memory_tuner/rlvram_revision_temporal.csv`): validation before training (step 0), every 20 steps and at the final step; checkpoint every 25 steps and at the final step. Screen: same schedule, 1 step → validation at 0 and 1, checkpoint at 1. Long run: validation 0, 20, 40, 60, 80, 100; checkpoints 25, 50, 75, 100. `train_max_samples=-1`, `val_max_samples=32`, `shuffle=False` (GSM8K: 3,200 distinct prompts; code: 64 training prompts → 2 steps/epoch → 50 fixed-order epochs, hence `total_epochs=100`, which is only the loop bound). `max_actor_ckpt_to_keep=3` (retains 50, 75, 100; GPU-memory neutral). **Instrumentation: external-only** (as the iteration-cost panel; avoids per-operation synchronization over 100 steps; the reservation-panel labels used full instrumentation, which the paper found label-neutral in eligible pairs).

**Restore**: VERL's V1 trainer already supports `trainer.resume_mode=resume_path` + `trainer.resume_from_path=…/global_step_50` (actor model/optimizer/extra shards, LR scheduler, RNG, dataloader state; LoRA-only and full state dicts) and re-synchronizes weights to vLLM in `on_init_end`. The new launcher passes `resume_from_path`, forces `del_local_ckpt_after_load=False`, and rejects implicit `auto` resume. The patch only marks the load as its own stage (`checkpoint_load`) and prints a parseable `RLVRAM_RESUME` line. The runner checks that the seed-631 long run's step-50 checkpoint is complete (data.pt and model/optim/extra shards of all 4 ranks), makes it read-only, records a SHA-256 manifest before and after the restore, and requires it unchanged. The restore continues to step 100 with `val_before_train=False` (validation 60, 80, 100; checkpoints 75, 100), i.e., the uninterrupted run's schedule after step 50. Trajectories are not bitwise identical (vLLM sampling RNG is reseeded); peaks, not trajectories, are compared. If no valid step-50 checkpoint exists, the restore slot stays `not_launched` with its reason.

Checkpoints go to `/scratch/project/fta-26-70/rlvram/checkpoints/<group>/<experiment>/attempt-<job>` (~32 GB each for 7B FP32 shards). Retention policy (frozen): after every invocation the runner writes `checkpoint-inventory.json`; at block end it deletes the block's own checkpoint directories (peak scratch ≈ 6 × 32 GB per allocation, ≈ 0.6 TB with three concurrent blocks).

Timeouts: screen 2,400 s; long 21,600 s; restore 12,600 s.

## 6. Seeds

Fresh: 601–603 (E1), 611–620 (E2), 631–633 (E3), 641 (smoke S1), 642 (smoke S2). `matrices/seed_audit.json` scans every matrix/profile CSV in R (training_seed/seed columns) and finds none of 590–700 used; earlier seeds were 0–2, 41–53, 61–96, 111–163, 211–243, 301–323, 441–473, 501–503. The MI250X panel reused A100 seeds.

## 7. Analysis plan and estimands (`ieee_resub/analysis.py`)

All differences are alternative minus reference; numeric contrasts use only complete blocks; every planned slot is reported with its state; nothing is replaced.

### 7.1 E1
Per block (complete = four eligible completions): for metric Y ∈ {actor-stage peak, whole-run peak} and logging level L ∈ {ext, full}: batch contrast Δ_L = Y(16,L) − Y(8,L); averaged Δ̄ = (Δ_ext+Δ_full)/2; logging interaction I = Δ_full − Δ_ext; per-token slope Δ_L / (max actor micro-batch tokens(16) − (8)) from the shape trace. **Primary pool**: complete blocks with identical shapes. Report per case n/mean/min/max; equal-case means over the four cases (only if all contribute, else contributing cases listed); side by side with the natural-length replication at 0.60 and the logging control; outcome counts per case; blocks where the actor stage sets the b16 whole-run peak. Predeclared diagnostics: D1 Δ_ext > 0 and Δ_full > 0 in every identical-shape block; D2 |I| ≤ 512 MiB in every identical-shape block (the 512 MiB threshold of SI S6); D3 |Δ̄ whole-run| ≤ 512 MiB in blocks where the actor stage does not set the peak.

### 7.2 E2
Per configuration: counts within/above/failure among eligible slots, unresolved, not launched; proportions with exact two-sided 95 % Clopper–Pearson intervals; ten-slot label; completed-peak min/median/max and minimum headroom; earlier outcomes reported alongside, not pooled.

### 7.3 E3
Per configuration: screen and long-run counts and repeated labels; individual screen→long agreement on within-margin approval; long minus screen peak; per long run step-1 peak, maximum later-step increase (max over steps ≥2 of the per-step peak minus step 1; per-step peaks include that step's validation/checkpoint samples), its step, validation and checkpoint peaks. Restore: restore whole-run peak (incl. checkpoint load and initial weight sync) minus the uninterrupted seed-631 run's peak over steps 51–100 (primary), restore steps-51–100 peak minus the same (secondary), restore minus uninterrupted whole-run peak, checkpoint-load stage peak.

### 7.4 Lifecycle checks used for completion
Checkpoint stage observed at every scheduled step; validation stage observed at every scheduled step; retained checkpoints (last three) present with data.pt and actor shards; `latest_checkpointed_iteration.txt` = final step; restore: resume line, `checkpoint_load` stage at step 50, source manifest unchanged, durable records exactly steps 51–100.

### 7.5 What would change which claim

| Result | Change in the paper |
|---|---|
| E1 D1 holds | Abstract/Results (lines 112, 1650–1663): the actor-stage increase persists with identical prompts, forced lengths and identical micro-batch composition; report identical-shape equal-case contrasts (ext/full) and per-token slopes next to 3,905 MiB; drop "generated responses can differ" as an explanation. 3,905 MiB stays the natural-length number. |
| E1 D1 fails in a case | Qualify the micro-batch claim for that case ("k of 4 cases"); the natural contrast there is not attributable to micro-batch alone. |
| E1 slopes within ±25 % of the natural-length slopes | State that the natural contrast scales with micro-batch tokens; length differences between conditions explain little. |
| E1 D2 holds | Lines 1778–1786: the 16 %/10 % logging differences reflect non-identical execution; allocator logging changes the identical-shape contrast by ≤ X MiB. |
| E1 D2 fails | Keep the caveat and report the identical-shape interaction as an instrumentation effect. |
| E1 actor stage sets the b16 whole-run peak (expected: Qwen/longer-prompt code) | Add: at forced cap lengths the whole-run peak no longer hides actor growth (as on MI250X); "hiding" depends on realized lengths. |
| E2 (a) < 10/10 within | Replace "three agreeing runs do not guarantee the same label" (line 1631) with the observed frequency and CI; any memory failure → release does not guarantee completion at 0.65. |
| E2 (b) any above margin | Line 1751 ("nine eligible runs with the same label … 194 MiB below"): report frequency; the label is not stable. 10/10 within → "19/19 within, one-sided 95 % upper bound 15 %". |
| E2 (c) any within | Above-margin label not deterministic; otherwise 19/19 above. |
| E2 (d) any above | Qualify the reservation-panel transition (0.60 within, 0.65 above) and interpret E3's Qwen offload-off results accordingly. |
| E3 all long runs within margin | Lines 2657–2671: 100-step runs with validation and checkpoints at 7B stay within margin; report max later growth. |
| E3 a within-margin screen pairs with an above-margin/failed long run | RQ3 Results and Discussion: one-step screens are insufficient for these 7B schedules; strengthen "test the intended schedule". |
| E3 restore − uninterrupted > 512 MiB or restore fails | Restore is a distinct memory risk; add to guidance. Otherwise report "restore within X MiB". |

All findings are reported regardless of direction; analyses not listed here are labeled exploratory.

## 8. Placement, software and storage on the cluster (FTA-26-70)

- Account `FTA-26-70` (OPEN-35-44 expired), partition `qgpu` (48 h limit), `--no-requeue`; the runner rejects allocations whose scheduler record is not FTA-26-70, not one node, not `Requeue=0`, or longer than the frozen wall cap.
- `/mnt/proj1/fta-26-70/rlvram/` (20 TB quota, empty today): `images/verl-vllm018-dev1.sif` (copied, SHA-256 verified, read-only), `python-packages/` (39 MB), `data/{gsm8k,math,codecontests/{heavy_tail,standard}}`, `runtime/kobzaond/cache/huggingface/hub/models--…` (five models, ≈47 GB), `venv/` (Anaconda3/2023.07-2 Python 3.11 + psutil 5.9.0, numpy 1.24.3, scipy 1.10.1, pytest 7.4.0, PyYAML), `src/ieee-resub-<commit12>/` (**immutable source snapshot**: clone of a git bundle of the approved commit, TransferQueue submodule at `5cb184e…`, clean, `chmod -R a-w`), `protocols/<study>/` (final protocols, gate files, submission receipts, all read-only), `output/<group>/` (raw evidence, block ledgers), `logs/`.
- `/scratch/project/fta-26-70/rlvram/`: `r/<job>` (runtime, Ray, caches), `r/port-reservations`, `checkpoints/` (E3).
- Inputs are copied from the still-readable OPEN-35-44 storage (valid to 2027-09-29) by `submit/setup_fta_storage.sh` without the old group; the final freeze recomputes every model/data/container hash and refuses to write if any differs from the earlier frozen records listed in `matrices/expected_inputs.json`.
- New code: `launcher/run_grpo_ieee_resub.slurm` (derived from `R/run_grpo_instrumented.slurm`; diff: account/paths, patched VERL copy on PYTHONPATH, period rule 1–99 for 2- and 4-GPU allocations, flags for fixed length/canonical order/shape trace, scratch checkpoints, explicit restore, extended trial record); `ieee_resub/runner.py` (from `memory_tuner/electronics_experiment.py`; 2- or 4-GPU allocations; all models verified; restore preparation; checkpoint release); `assess.py`, `collect.py` (from `collect_electronics.py`), `analysis.py`, `plan.py`, `render.py`, `env.py`; `slurm/ieee_resub_<study>.slurm`; `submit/submit_ieee_resub.sh`.

## 9. Amendment and failure policy
As in the earlier panels: one permanent claim per block (directory creation), no automatic retries/requeues/replacements, continue in frozen order only if the cleanup/idle gates pass and the remaining allocation covers the next period's timeout + 60 s; otherwise remaining periods are `not_launched`. A rerun requires a written amendment (`protocols/<study>/amendment-NN.json`, hash-chained to the protocol) **before** the rerun, documenting a logged non-memory defect (infrastructure, harness) and listing the affected slots; diagnosed memory outcomes and above-margin completions are never rerun; original attempts stay in the ledger. Changes of cases or settings after S1/S2 but before the formal freeze are recorded as pre-freeze amendments to this draft.

## 10. Smoke tests (development only; excluded from all estimates)

- **S1** (`smoke1`, seed 641, one 2-GPU allocation, `--time=01:30:00`): Granite/GSM8K b16-full, Granite/GSM8K b08-ext, Qwen-3B/longer-prompt code b16-full. Pass criteria: all eligible completions; every response length = cap; identical shape signatures for the two Granite runs; canonical-order flag and shape trace valid; Qwen b16 actor-stage peak ≤ 37,000 MiB (else pre-freeze amendment: replace that case by Qwen-3B/standard code, predicted 30.4 GB).
- **S2** (`smoke2`, seed 642, one 4-GPU allocation, `--time=02:00:00`): Qwen-7B/GSM8K offload on, 0.70: 4 steps (validation 0, 2, 4; checkpoints 2, 4; keep 3), then restore from step 2 to step 4. Pass: completion; lifecycle checks; `RLVRAM_RESUME` line; `checkpoint_load` stage; durable records 3–4; source manifest unchanged; collector and analysis run on the real outputs.
- The user records the review as `protocols/smoke1/smoke1_gate.json` / `smoke2_gate.json`; `submit_ieee_resub.sh` refuses E1 without S1, E2 without S1+S2, E3 without S2.

## 11. Execution sequence after approval
- **S0**: commit this directory; `git bundle create` locally; copy the bundle with scp to `/mnt/proj1/fta-26-70/rlvram/src/bundles/` (no git remote); run `submit/setup_fta_storage.sh <commit> <bundle> --execute` on a login node (layout, copies, venv, snapshot, `plan.py freeze-final` for all five studies → protocol SHA-256s).
- **S1/S2**: `submit_ieee_resub.sh smoke1|smoke2 <protocol> <sha> <commit> --submit`; `collect.py` + `analysis.py`; write gate files.
- **S3**: submit E3 (longest), E1, E2 (`dryrun/submit_commands.sh` lists the exact commands).
- **S4**: `collect.py` per study (raw reassessment; must equal the terminal receipts) → `analysis.py e1|e2|e3`.
- **S5**: archive raw evidence with SHA-256 manifest, copy results back to this directory, add Table 6 rows and SI S23 entries.

## 12. Cost estimate (Karolina qgpu: 8 A100 per node; accounted node-hours = GPU-hours/8, as `billing` = GPU count in the earlier `sacct` records; verify with `it4ifree` after S1/S2)

| Study | Basis | Expected GPU-h | Expected NH | Cap (all walls used) GPU-h / NH |
|---|---|---:|---:|---:|
| S1+S2 | 3 × 2-GPU 1-step runs; 4+2 7B steps | 3 | 0.4 | 11 / 1.4 |
| E1 | replication blocks took 2,000–2,200 GPU-s (2 GPUs); +40 % for forced lengths → ≈1,500 s × 2 GPUs × 12 | 10 | 1.3 | 48 / 6 |
| E2 | release panel ≈273 s/invocation, reservation panel ≈392 s (7B); ≈1,230 s/block × 4 GPUs × 10 | 14 | 1.7 | 67 / 8.3 |
| E3 | iteration-cost panel steady step time: Qwen off1 r70 81–89 s, Qwen off0 62–122 s, Mistral off1 r70 94–135 s, Mistral off0 72–92 s; +6 validations, 4 checkpoints, startup ≈880 s per long run; screens ≈0.16 h; restores ≈1.5 h | 162 | 20.3 | 552 / 69 |
| **Total** | | **≈190** | **≈24** | **678 / 85** |

With a 1.5× slow-node contingency (one Qwen block ran at 122 s/step): ≈285 GPU-h ≈ 36 NH, 7 % of the 500 NH allocation; the hard cap is 17 %. Wall-clock: E3 dominates (12 blocks, 3 concurrent, 2.5–4 h each, restore blocks ≈5–6 h) ≈ 16–20 h after queueing.

## 13. Local dry run and tests (10 November 2026)
- `python -I -B ieee_resub/plan.py build`: 5 matrices (E1 48, E2 40, E3 28, S1 3, S2 2 = 121 invocations), drafts `matrices/protocol_draft/*.json` (status `draft-awaiting-approval`), seed audit (fresh), expected input hashes from earlier frozen records.
- `python -I -B ieee_resub/render.py dryrun`: every one of the 121 invocations executed the real launcher in a sandbox (stub python/apptainer; only the two FTA path assertions neutralized); 0 problems: launcher exit 0, trial record matches the matrix for 28 fields, every Hydra override key exists in the patched VERL config, E1 flags reach the container, wall caps cover the worst case of every block. Outputs: `dryrun/sbatch/*.slurm`, `dryrun/invocations/<study>.jsonl` (environment, container options and trainer command per invocation, sandbox paths relabeled to cluster paths), `dryrun/submit_commands.sh`, `dryrun/report.json`. **0 jobs submitted.**
- `python -I -B ieee_resub/analysis.py check-existing`: on the archived outputs it reproduces 3,904.6 → **3,905 MiB** and 7.46 → **7 MiB** (35 complete blocks, 12 cases; block range 773–9,034), 4,060 MiB reservation contrast; logging control 1,966/6,430 (external) and 2,287/7,059 (full) MiB, interaction −220…1,054 MiB with 3 blocks beyond 512, whole-run −160…132 MiB; paired-duration later growth 159/657/115/590 MiB and 10 individual agreements; release anchor 2/3 within, 95 % CI 0.094–0.992; one-sided bounds 63 %/28 %.
- `pytest tests/` (36 tests): shape-trace helpers, matrix balance and provenance, launcher sandbox (E1/E2/E3/restore, rejection of implicit resume and missing checkpoints), assessment units (step peaks, growth, resumed step ranges, lifecycle, restore evidence, 2-GPU identity), analysis, patch reproducibility, runner restore/release guards.

## 14. Needs approval
1. E1 case changes (Phi/MATH and Qwen-3B/MATH instead of Phi/longer-prompt code and Qwen-3B/GSM8K), forcing to the full response cap, and canonical batch ordering (E1 only).
2. E2 full instrumentation and 4-GPU allocations with 2 task GPUs for (a)–(c).
3. E3 external-only instrumentation, schedule (validation 0/20/…/100, checkpoints 25/50/75/100, keep 3), `total_epochs=100`, restore design, checkpoint deletion at block end.
4. Seeds, order seeds, timeouts and wall caps; budget (≈24 NH expected, ≤85 NH cap).
5. Storage layout and copying ≈60 GB of inputs into proj1; the commit + bundle transfer; the smoke gates.
6. The VERL patch (opt-in; defaults unchanged) and the new launcher.
