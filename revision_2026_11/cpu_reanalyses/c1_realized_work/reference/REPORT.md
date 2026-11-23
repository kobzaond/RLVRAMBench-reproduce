# C1. Realized work behind the micro-batch contrast (reviewer R1.3)

CPU-only re-analysis of existing measurements; nothing was run on GPUs or clusters, nothing was pushed or uploaded, and no manuscript file was edited. All numbers below are produced by `c1_realized_work.py` in this directory (run with `python -I`).

## 1. Data sources

Repository root R = `/home/kobzaond/Dokumenty/alquictCoder/research/ieee_access/rl`.

| Content | Path |
|---|---|
| Replication slots, status, stage peaks (144 rows) | `R/benchmark/replication_control/results/replication_control_trials.csv` |
| Replication configuration matrix (micro-batch, reservation, caps) | `R/benchmark/replication_control/replication_control_matrix.csv` |
| Replication per-block contrasts (cross-check) | `R/benchmark/replication_control/results/replication_control_blocks.csv` |
| Replication equal-case summary (cross-check) | `R/benchmark/replication_control/results/replication_control_summary.json` |
| Replication reservation-specific pair contrasts (cross-check) | `R/profiles/electronics_revision/stage_paired_contrasts.csv` |
| Original factorial trial table (job ids, settings) | `R/profiles/standard_grpo/factorial-trials.csv` |
| Original factorial stage peaks (actor NVML, whole run) | `R/profiles/standard_grpo/factorial-phase-trials.csv` |
| Original factorial per-block contrasts (cross-check) | `R/profiles/standard_grpo/factorial-phase-blocks.csv` |
| Original factorial realized workload (cross-check of log parsing) | `R/profiles/review_revision/realized-workload-trials.csv` |

Per-process trainer metrics are not in the working tree; they exist only in the released archive. The archive was downloaded once into a fresh `mktemp -d` directory and verified before use:

- URL: `https://huggingface.co/datasets/kobzaond/RLVRAMBench/resolve/00e61d8aa329cee64c1cf899f11678971b5115e0/releases/RLVRAMBench-review-artifact-submission-20260917-b0018d0.tar.zst`
- Local copy: `/tmp/c1_archive.4VRRkr/RLVRAMBench-review-artifact-submission-20260917-b0018d0.tar.zst` (210,126,067 bytes)
- SHA-256: `3dd90eaca2f2d9b5eceb955fe67b6a3528210a6bb9bd7684ac3eb3c36d84a5ab` (matches the expected value and `R/paper/electronics_delivery_manifest.json`)
- Extracted members only (into `/tmp/c1_archive.4VRRkr/extract`): the 144 files `output/replication_control/attempts/<experiment_id>/trainer-metrics/step-1.json` and the 144 retained original-factorial logs `output/strengthening_factorial/<experiment_id>/training-<job_id>.log` (job ids from `R/profiles/standard_grpo/factorial-trials.csv`). Nothing from the archive was executed.

Each `step-1.json` holds the trainer's step-1 console line; the script checks that its `invocation_id` equals the attempt directory and that `console_line_sha256` matches the line.

**Definitions.**

- *Total tokens* = `perf/total_num_tokens` (prompt plus response tokens over all sequences of the step; 64 sequences = 32 prompts x 2 responses, MATH 128). The script verifies total = sequences x (prompt mean + response mean) and total = 2 x `global_seqlen/mean` for every process.
- *Per-GPU tokens* = `global_seqlen/mean` (total/2) and `global_seqlen/balanced_max` (larger of the two per-GPU token loads after verl's sequence balancing).
- *Cap reaching* follows `extract_workload_log` in `R/memory_tuner/review_attempt_workload_audit.py`: a process reaches the cap when `response_length/max` >= configured `max_response_length`. `response_length/clip_ratio` is the share at the padded batch maximum, so it is the share at the cap only when the maximum reaches the cap; the number of responses at the cap is computed as clip_ratio x sequences for cap-reaching processes and 0 otherwise. The logged clip ratios are kept unchanged in `rc_attempts.csv`.
- *Actor-stage peak* = `actor_update_peak_mib` (replication) or `actor_nvml_mib` (original factorial): the largest NVML sample on either GPU during the actor-update interval. *Whole-run peak* = whole-process peak.
- *Pair* = the two conditions with micro-batch 16 and 8 at one reservation in one complete block (70 pairs in the replication, 72 in the original factorial). The block batch contrast is the mean of its two pairs, so equal-case means over pairs equal the published equal-case means.

## 2. Reproduction check (passed)

| Check | Computed | Reference | OK |
|---|---|---|---|
| replication complete blocks | 35 | 35 | yes |
| replication completed processes | 140 | 140 | yes |
| replication equal-case actor-stage batch contrast (MiB) | 3904.6111 | 3904.6111 | yes |
| replication equal-case whole-run batch contrast (MiB) | 7.4583 | 7.4583 | yes |
| replication actor-stage equal-case mean, rounded | 3905 | 3905 | yes |
| replication whole-run equal-case mean, rounded | 7 | 7 | yes |
| replication 12 case rows of tab:factorial (actor, whole-run) | all match | 12 rows | yes |
| replication per-block batch contrasts vs released blocks CSV | all 35 match | benchmark/replication_control/results/replication_control_blocks.csv | yes |
| replication 70 reservation-specific pair contrasts vs stage_paired_contrasts.csv | all 70 match | profiles/electronics_revision/stage_paired_contrasts.csv | yes |
| original factorial complete blocks | 36 | 36 | yes |
| original factorial per-block actor-stage contrasts vs factorial-phase-blocks.csv | all 36 match | profiles/standard_grpo/factorial-phase-blocks.csv | yes |
| original factorial equal-case actor-stage batch contrast, rounded | 3980 | 3980 | yes |
| original factorial equal-case whole-run batch contrast, rounded | -31 | -31 | yes |

The unadjusted equal-case means reproduce the manuscript exactly: actor stage 3904.61 MiB -> 3,905 MiB and whole run 7.46 MiB -> 7 MiB (35 complete blocks, 140 processes); all twelve case rows of Table tab:factorial match (half-up rounding; Qwen/standard code whole run is 66.5 -> 67). The original factorial gives 3980.06 -> 3,980 MiB and -30.61 -> -31 MiB. Individual replication block contrasts range from 773 to 9,034 MiB, as in the text.

## 3. Key results: same-device replication

**Prompts.** Prompt-length statistics (mean, max, min) are identical across all conditions and all seeds of each case (the launch script uses `data.shuffle=False`, and each process trains one step on the first batch). Token differences between conditions are therefore entirely response-token differences.

**Cap reaching.** 79 of 140 completed processes have at least one response at the cap; by condition (MB8/0.60, MB16/0.60, MB8/0.70, MB16/0.70): 19, 20, 21, 19 of 35. Share of responses at the cap: 11.5%, 11.2%, 11.4%, 11.0% (pooled over 2,816 responses per condition). Cap reaching is concentrated in MATH (cap 512; every process of every model), Phi standard and longer-prompt code, and Qwen longer-prompt code; by case: Granite/GSM8K 5; Granite/MATH 12; Granite/Standard code 0; Granite/Longer-prompt code 0; Phi/GSM8K 2; Phi/MATH 12; Phi/Standard code 11; Phi/Longer-prompt code 12; Qwen2.5-3B/GSM8K 0; Qwen2.5-3B/MATH 12; Qwen2.5-3B/Standard code 3; Qwen2.5-3B/Longer-prompt code 10 (processes of 12; Qwen standard code of 8).

**Token volume by condition.** Equal-case mean total tokens per process: MB8/0.60 50,281, MB16/0.60 50,324, MB8/0.70 50,584, MB16/0.70 50,137. Per case x condition values (means and ranges over seeds) are in `rc_case_condition_summary.csv`; per-process values in `rc_attempts.csv`.

**Within-pair token differences (MB16 - MB8), n = 70 pairs.** Relative total-token difference (T16 - T8)/T8: median -0.17%, IQR -1.80% to 1.70%, range -10.9% to 12.1%, mean -0.23%, mean absolute 2.57%. Pairs within 1% / 2% / 5% / 10%: 24 / 38 / 61 / 67. MB16 has more tokens in 32 pairs and fewer in 38. Response tokens alone: median -0.40%, range -20.0% to 18.6%, mean absolute 5.11%. Block-level (averaged over reservations, n = 35): median -0.56%, range -5.9% to 6.6%. Reference: between the two reservations at a fixed micro-batch (70 pairs), the mean absolute difference is 2.66% (range -16.7% to 11.6%), i.e. the micro-batch pairs differ in realized work by about as much as any two conditions of a block. Per-case ranges: `rc_case_token_differences.csv`.

**Sign of the actor-stage contrast.** The actor-stage peak is higher at MB16 in all 70 of 70 pairs (range 470 to 9,510 MiB), including all 38 pairs in which MB16 processed fewer tokens (smallest contrast there: 470 MiB).

**Correlation (Spearman).** Pooled across cases, the relative token difference and the pair actor-stage contrast are weakly related (rho = 0.18, n = 70), because case differences dominate the contrast. After centering both quantities within case, rho = 0.71 (absolute tokens) and 0.73 (relative tokens); per-GPU balanced-max tokens 0.73; longest response 0.41; at block level (n = 35, centered) 0.70. For the whole-run contrast, rho = 0.16. Thus realized work explains a substantial part of the block-to-block spread of the actor-stage contrast within a case, not its sign or mean. (p-values in `rc_correlations.csv` are descriptive only; pairs share blocks.)

**Contrast restricted to token-matched pairs** (equal-case mean over cases with at least one kept pair):

| Restriction | Pairs kept | Cases | Equal-case actor-stage contrast (MiB) | Same cases, all pairs (MiB) | Whole-run contrast, kept (MiB) | Kept pairs per case |
|---|---|---|---|---|---|---|
| all pairs | 70 / 70 | 12 | 3,905 | 3,905 | 7.5 | Granite/GSM8K:6; Granite/MATH:6; Granite/Standard code:6; Granite/Longer-prompt code:6; Phi/GSM8K:6; Phi/MATH:6; Phi/Standard code:6; Phi/Longer-prompt code:6; Qwen2.5-3B/GSM8K:6; Qwen2.5-3B/MATH:6; Qwen2.5-3B/Standard code:4; Qwen2.5-3B/Longer-prompt code:6 |
| &#124;rel total-token diff&#124; < 5% | 61 / 70 | 12 | 4,040 | 3,905 | 8.5 | Granite/GSM8K:3; Granite/MATH:6; Granite/Standard code:6; Granite/Longer-prompt code:6; Phi/GSM8K:4; Phi/MATH:6; Phi/Standard code:4; Phi/Longer-prompt code:4; Qwen2.5-3B/GSM8K:6; Qwen2.5-3B/MATH:6; Qwen2.5-3B/Standard code:4; Qwen2.5-3B/Longer-prompt code:6 |
| &#124;rel total-token diff&#124; < 2% | 38 / 70 | 12 | 4,063 | 3,905 | 6.1 | Granite/GSM8K:2; Granite/MATH:5; Granite/Standard code:4; Granite/Longer-prompt code:5; Phi/GSM8K:1; Phi/MATH:3; Phi/Standard code:2; Phi/Longer-prompt code:1; Qwen2.5-3B/GSM8K:3; Qwen2.5-3B/MATH:6; Qwen2.5-3B/Standard code:1; Qwen2.5-3B/Longer-prompt code:5 |
| &#124;rel total-token diff&#124; < 1% | 24 / 70 | 10 | 3,739 | 3,634 | 15.7 | Granite/GSM8K:1; Granite/MATH:3; Granite/Standard code:4; Granite/Longer-prompt code:3; Phi/GSM8K:0; Phi/MATH:2; Phi/Standard code:2; Phi/Longer-prompt code:0; Qwen2.5-3B/GSM8K:2; Qwen2.5-3B/MATH:5; Qwen2.5-3B/Standard code:1; Qwen2.5-3B/Longer-prompt code:1 |
| &#124;rel response-token diff&#124; < 5% | 45 / 70 | 11 | 3,629 | 3,524 | 9.2 | Granite/GSM8K:3; Granite/MATH:6; Granite/Standard code:4; Granite/Longer-prompt code:3; Phi/GSM8K:3; Phi/MATH:6; Phi/Standard code:3; Phi/Longer-prompt code:0; Qwen2.5-3B/GSM8K:6; Qwen2.5-3B/MATH:6; Qwen2.5-3B/Standard code:1; Qwen2.5-3B/Longer-prompt code:4 |
| &#124;rel response-token diff&#124; < 2% | 22 / 70 | 10 | 3,731 | 3,634 | 16.8 | Granite/GSM8K:1; Granite/MATH:4; Granite/Standard code:2; Granite/Longer-prompt code:2; Phi/GSM8K:0; Phi/MATH:2; Phi/Standard code:2; Phi/Longer-prompt code:0; Qwen2.5-3B/GSM8K:2; Qwen2.5-3B/MATH:5; Qwen2.5-3B/Standard code:1; Qwen2.5-3B/Longer-prompt code:1 |

**Within-case adjustment.** Per case, OLS of the actor-stage peak on an MB16 indicator, a reservation-0.70 indicator, block fixed effects, and total tokens (12 processes and 6 residual df per case; Qwen standard code 8 and 3). Without the token term the MB16 coefficient equals the published case contrast. Equal-case mean MB16 coefficient: unadjusted 3,905 MiB; adjusted for total tokens 3,970 MiB; for per-GPU balanced-max tokens 3,963 MiB; for the longest response 3,879 MiB. Per-case change from token adjustment: -85 to 481 MiB (-6.2% to 9.8%); token slopes range from -75 to 830 MiB per 1,000 tokens. Per-case coefficients and OLS standard errors: `rc_case_adjustment.csv`. The adjustment assumes one token slope for both micro-batches and is descriptive (small n).

**Unresolved block.** The four unresolved slots (Qwen standard code, seed 211; collector reason `Trial scheduler/exit identity differs`) also logged step 1; their total tokens are joint 45,707, base 46,389, actor_only 45,118, reservation_only 45,404. They remain outside all contrasts and are listed with `scientific_eligible = 0` in `rc_attempts.csv`.

## 4. Key results: original factorial (separate allocations, 144 processes, 36 blocks)

Token data come from the retained training logs (`perf/total_num_tokens` etc.); the parsed prompt and response means and cap flags match `realized-workload-trials.csv` for all 144 processes. Prompts are again identical within each case. Processes reaching the cap: 81 of 144 (by condition 21, 20, 18, 22 of 36); responses at the cap: 11.0%, 11.2%, 10.7%, 10.5%. Relative token difference over 72 pairs: median 0.20%, range -13.4% to 11.0%, mean absolute 2.36% (reservation reference 2.62%); within 2% / 5%: 43 / 62. All 72 pair contrasts are positive (764 to 9,790 MiB), including the 33 pairs with fewer MB16 tokens. Within-case Spearman rho = 0.60 (pooled 0.06). Restricted equal-case means: <5% 3,982 MiB (62 pairs, 12 cases); <2% 4,095 MiB (43 pairs, 12 cases). Token-adjusted equal-case coefficient 4,045 MiB versus 3,980 MiB unadjusted (per-case change -131 to 433 MiB). Files: `of_*.csv`.

## 5. Interpretation and limits

- Realized work does differ between conditions with matched seeds: the mean absolute difference is 2.6% of total tokens, and single pairs differ by up to 13.4% of total tokens (25.1% of response tokens) across both experiments. The differences have no systematic direction (median -0.17%; 32 positive and 38 negative) and are as large between the two reservations at a fixed micro-batch (2.7% mean absolute), which is consistent with run-to-run variation in generation. In the launch configuration the varied setting is `actor.ppo_micro_batch_size_per_gpu`, which applies to the actor update after generation in the single measured step; the old- and reference-log-prob micro-batches are fixed at 4 in all conditions.
- Within a case, token differences track the size of the pair contrast (Spearman rho 0.71 in the replication, 0.60 in the original factorial), so part of the block-to-block range of the contrast reflects realized work. The mean contrast is insensitive: the token-matched subsets (5%, 2%, 1%) and the three adjusted fits change the equal-case mean by -0.7% to 4.0% across both experiments (restricted means compared with all pairs of the same contributing cases).
- The trainer logs only step-level aggregates. Tokens per actor micro-batch, which most directly drive actor-update activations, are not recorded; the per-GPU balanced-max load and the longest response are the closest available proxies and lead to the same conclusion.
- The 2% and 5% thresholds are post hoc. At 2%, some cases keep at most one pair (Phi / GSM8K; Phi / Longer-prompt code; Qwen2.5-3B / Standard code), so per-case restricted means are noisy; the regression has 3-6 residual degrees of freedom per case and assumes one token slope for both micro-batches.
- The wording "observed configuration contrast" therefore remains appropriate: tokens differ materially in some pairs, even though the mean contrast does not depend on them.

## 6. Proposed manuscript text

Because generated responses can differ between conditions despite matched seeds, Table S[X] reports the realized work of every replication condition, and the released per-process records give prompt and response lengths, cap reaching, and token volumes. Prompt lengths are identical across all conditions and seeds of a case, because each process trains one step on the same unshuffled first batch; token volumes therefore differ only through the generated responses. In each of the four conditions, between 11.0% and 11.5% of the responses reach the length cap. Across the 70 reservation-specific micro-batch pairs of the 35 complete blocks, the total token volume at micro-batch 16 differs from that at micro-batch 8 by a median of -0.2% (range -10.9% to 12.1%), about as much as the two reservations differ at a fixed micro-batch (mean absolute difference 2.6% versus 2.7%). Within a case, larger token differences accompany larger actor-stage contrasts (Spearman ρ = 0.71), yet the actor-stage peak is higher at micro-batch 16 in all 70 pairs, including the 38 pairs in which the micro-batch-16 process handled fewer tokens. Restricting the comparison to the 38 pairs whose token volumes differ by less than 2%, which cover all twelve cases, gives an equal-case mean of 4,063 MiB (4,040 MiB for the 61 pairs below 5%), and a within-case fit with block fixed effects and total tokens as a covariate gives 3,970 MiB, compared with 3,905 MiB without adjustment. The original factorial experiment shows the same pattern (3,980 MiB unadjusted, 4,095 MiB in the 43 pairs within 2%, and 4,045 MiB after adjustment). The 3,905 MiB value therefore remains an observed configuration contrast under full instrumentation: realized work varies by a few percent between conditions and contributes to the spread between blocks, but it does not account for the mean increase.

LaTeX-ready version (same content):

```latex
Because generated responses can differ between conditions despite matched seeds, Table~S[X] reports the realized work of every replication condition, and the released per-process records give prompt and response lengths, cap reaching, and token volumes. Prompt lengths are identical across all conditions and seeds of a case, because each process trains one step on the same unshuffled first batch; token volumes therefore differ only through the generated responses. In each of the four conditions, between 11.0\% and 11.5\% of the responses reach the length cap. Across the 70 reservation-specific micro-batch pairs of the 35 complete blocks, the total token volume at micro-batch 16 differs from that at micro-batch 8 by a median of \(-0.2\)\% (range \(-10.9\)\% to 12.1\%), about as much as the two reservations differ at a fixed micro-batch (mean absolute difference 2.6\% versus 2.7\%). Within a case, larger token differences accompany larger actor-stage contrasts (Spearman \(\rho = 0.71\)), yet the actor-stage peak is higher at micro-batch 16 in all 70 pairs, including the 38 pairs in which the micro-batch-16 process handled fewer tokens. Restricting the comparison to the 38 pairs whose token volumes differ by less than 2\%, which cover all twelve cases, gives an equal-case mean of 4,063~MiB (4,040~MiB for the 61 pairs below 5\%), and a within-case fit with block fixed effects and total tokens as a covariate gives 3,970~MiB, compared with 3,905~MiB without adjustment. The original factorial experiment shows the same pattern (3,980~MiB unadjusted, 4,095~MiB in the 43 pairs within 2\%, and 4,045~MiB after adjustment). The 3,905~MiB value therefore remains an observed configuration contrast under full instrumentation: realized work varies by a few percent between conditions and contributes to the spread between blocks, but it does not account for the mean increase.
```

## 7. Proposed Supporting Information table (booktabs)

Compact table (also written to `si_table_compact.tex`):

```latex
\begin{table}[!htbp]
\centering
\caption{Realized work behind the micro-batch contrast in the same-device replication.
Each micro-batch pair compares batch 16 with batch 8 at one reservation within one complete
block; \(n\) is the number of pairs. Prompts are identical across conditions within a case,
so token differences arise from generated responses. Tokens are prompt plus response tokens
per process (mean over pairs). Responses at cap: share of generated responses whose length
equals the configured response cap. Token difference: range over pairs of
\((T_{16}-T_{8})/T_{8}\). Actor-stage contrast: mean pair difference in MiB (as in
Table~\ref{tab:factorial}). Token-adjusted: micro-batch coefficient of a within-case
least-squares fit with block fixed effects, a reservation indicator, and total tokens.
Pairs within 2\%: pairs with \(|T_{16}-T_{8}|/T_{8}<0.02\) and their mean contrast.
Summary rows: pooled cap shares, the overall token-difference range, and equal-case means
of the contrasts (twelve cases weighted equally); the original factorial row uses the 36
complete blocks of the separate-allocation experiment.}
\label{tab:si-realized-work}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrcrrrr@{}}
\toprule
 & & & \multicolumn{2}{c}{Tokens (\(\times 10^3\))} & \multicolumn{2}{c}{At cap (\%)} & Token diff. & Actor-stage & Token- & \multicolumn{2}{c}{Pairs within 2\%} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){11-12}
Model & Workload & \(n\) & 8 & 16 & 8 & 16 & 16 vs 8 (\%) & contrast & adjusted & \(n\) & Mean \\
\midrule
Granite & GSM8K & 6 & 27.9 & 28.3 & 1.0 & 0.8 & \(-10.9\) to 12.1 & 1,374 & 1,288 & 2 & 1,419 \\
 & MATH & 6 & 65.4 & 64.9 & 27.7 & 28.5 & \(-2.1\) to 1.3 & 1,932 & 1,894 & 5 & 1,876 \\
 & Standard code & 6 & 46.5 & 45.8 & 0.0 & 0.0 & \(-4.8\) to 1.0 & 2,067 & 2,185 & 4 & 2,158 \\
 & Longer-prompt code & 6 & 70.2 & 69.9 & 0.0 & 0.0 & \(-1.6\) to 2.3 & 3,460 & 3,491 & 5 & 3,415 \\
\addlinespace[3pt]
Phi & GSM8K & 6 & 19.5 & 19.5 & 0.3 & 0.3 & \(-5.8\) to 10.8 & 2,424 & 2,489 & 1 & 2,558 \\
 & MATH & 6 & 56.5 & 56.6 & 21.2 & 20.6 & \(-4.0\) to 2.9 & 4,635 & 4,609 & 3 & 4,724 \\
 & Standard code & 6 & 46.0 & 45.0 & 6.0 & 4.2 & \(-7.6\) to 3.7 & 4,915 & 5,396 & 2 & 5,814 \\
 & Longer-prompt code & 6 & 69.1 & 68.0 & 9.9 & 8.9 & \(-8.1\) to 3.6 & 8,093 & 8,491 & 1 & 8,558 \\
\addlinespace[3pt]
Qwen2.5-3B & GSM8K & 6 & 25.8 & 26.1 & 0.0 & 0.0 & \(-2.5\) to 3.4 & 2,682 & 2,607 & 3 & 2,620 \\
 & MATH & 6 & 63.8 & 63.7 & 25.0 & 24.6 & \(-1.8\) to 0.7 & 3,723 & 3,735 & 6 & 3,723 \\
 & Standard code & 4 & 45.6 & 45.9 & 0.4 & 0.8 & \(-4.3\) to 3.1 & 4,942 & 4,915 & 1 & 5,400 \\
 & Longer-prompt code & 6 & 68.8 & 69.2 & 2.6 & 1.3 & \(-1.5\) to 3.6 & 6,608 & 6,543 & 5 & 6,486 \\
\midrule
All cases & & 70 & & & 11.5 & 11.1 & \(-10.9\) to 12.1 & 3,905 & 3,970 & 38 & 4,063 \\
Original factorial & & 72 & & & 10.8 & 10.8 & \(-13.4\) to 11.0 & 3,980 & 4,045 & 43 & 4,095 \\
\bottomrule
\end{tabular*}
\end{table}
```

A per-condition long table (48 rows: every case x condition with response mean/range, longest response, cap reaching, token volume, and mean peaks; requires `longtable`) is written to `si_table_per_condition.tex`; it answers the request for every replication condition literally if the editor prefers it.

## 8. Output files

- `c1_realized_work.py`: analysis script (stdlib + numpy + scipy; run with python -I)
- `reproduction_check.csv`: all reproduction checks
- `key_results.json`: machine-readable key numbers
- `rc_attempts.csv / of_attempts.csv`: per process: case, seed, block, condition, lengths, cap reaching, tokens (total, per GPU), stage peaks
- `rc_case_condition_summary.csv / of_case_condition_summary.csv`: per case x condition means and ranges
- `rc_block_pairs.csv / of_block_pairs.csv`: per block x reservation: MB16 - MB8 token, length, cap and peak differences
- `rc_blocks.csv / of_blocks.csv`: per block: reservation-averaged batch contrasts (peaks and tokens)
- `rc_token_difference_distribution.csv / of_...`: distribution of relative token differences, incl. reservation reference
- `rc_case_token_differences.csv / of_...`: per-case token-difference ranges and contrast ranges
- `rc_correlations.csv / of_...`: Spearman correlations (pooled and within-case)
- `rc_restricted_overall.csv / rc_restricted_by_case.csv / of_...`: token-matched restricted contrasts
- `rc_case_adjustment.csv / of_...`: per-case OLS: unadjusted vs token-adjusted micro-batch coefficient
- `si_table_compact.tex, si_table_per_condition.tex`: proposed SI tables
