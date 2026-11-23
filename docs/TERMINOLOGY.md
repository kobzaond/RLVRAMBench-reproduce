# Terminology: operating limit, `safe`, and related fields

This file supplements `DATA_DICTIONARY.md`, which is hash-bound to the historical archive
(`source-provenance.json`) and therefore unchanged in this repository. The dataset's
`DATA_DICTIONARY.md` (release `rlvrambench-2026-11`) contains the same section.

## Operating limit and the shorthand `safe`

**Operating limit.** Every label uses one declared operating convention:
device capacity minus 2,048 MiB. On the 40,960 MiB A100-SXM4-40GB this is
**38,912 MiB** (inclusive); on a 65,520 MiB MI250X GCD it is 63,472 MiB.
Memory is the largest sampled used-memory value on any one allocated device
(external NVML or ROCm sampling at 100 ms), not a sum across devices. The
limit is a margin chosen before the measurements. It is **not** a physical
memory threshold, **not** a calibrated failure probability, and **not** a
guarantee that a configuration will complete in a future run.

**`safe` and `unsafe`.** In field names, `safe` is shorthand for
*within-margin completion only*: the process (or, for a configuration, every
eligible repetition) completed the requested run and its sampled per-device
peak stayed at or below the operating limit. `unsafe` means "not within
margin": it covers both completions above the limit (which did complete) and
diagnosed memory failures. Neither term denotes a physical threshold, a
failure probability, or guaranteed completion. A `safe` configuration can
fail in a later run, an `unsafe` above-margin configuration completed, and a
missing (blank) value means unresolved or ineligible evidence, not `0`.
The three-state labels `within_margin`, `above_margin`, and `memory_failure`
carry the same information without the shorthand and are preferred.

| Field | Meaning |
|---|---|
| `safe` | Process level: `1` if the process completed the requested run and its peak is at or below the operating limit; `0` if it completed above the limit or had a diagnosed memory failure; blank if not eligible. Configuration or cell level: `1` only if every eligible repetition is `safe`. |
| `safe_repetitions` | Number of eligible repetitions with `safe = 1`. |
| `<condition>_safe` | `safe` of the named condition of a matched comparison, for example `base_safe`, `actor_only_safe`, `reservation_only_safe`, `joint_safe`, `external_safe`, `full_safe`, `two_gpu_safe`, `four_gpu_safe`, `source_safe`, `long_safe`. |
| `actor_main_effect_safe`, `reservation_main_effect_safe`, `interaction_safe` | Factorial contrasts of the 0/1 `safe` indicators within one block (main effects averaged over the other factor; interaction = joint − actor-only − reservation-only + base). Descriptive, not probabilities. |
| `safe_difference` | Second minus first condition's `safe` indicator in a pair (−1, 0, or 1). |
| `safety_discordance` | `1` if the two conditions of a pair have different `safe` indicators. |
| `safe_targets`, `unsafe_targets` | Number of target configurations that are within margin, and that are not (above margin or memory failure). |
| `predicted_safe` | `1` if the transfer rule approves the target, that is, predicts within-margin completion. |
| `false_safe_count` | Approved targets that are not within margin (approved memory failures plus approved above-margin completions). |
| `false_unsafe_count` | Within-margin targets that the rule rejects. |
| `false_safe_rate`, `false_unsafe_rate` | `false_safe_count / unsafe_targets` and `false_unsafe_count / safe_targets` (blank when the denominator is zero). Descriptive error shares on the listed targets, not failure probabilities for future jobs. |
| `two_to_four_false_safe`, `two_to_four_false_unsafe`, `four_to_two_false_safe`, `four_to_two_false_unsafe` | GPU-count transfer: `1` if copying the source GPU count's `safe` label approves a target that is not within margin (`false_safe`) or rejects a within-margin target (`false_unsafe`). |
| `highest_safe_level`, `<model>_max_safe_level` | Highest configuration level (0–5 for c0–c5) whose configuration is within margin for that case; `-1` if none is. |
| `safe_configurations` | Number of within-margin configurations. |

**Temporal-safety fields** (one-step screens versus longer runs):

| Field | Meaning |
|---|---|
| `source_one_step_safe` | `safe` of the one-step screen used as the source label. |
| `long_run_safe` | `1` if the longer run completed every requested step and its whole-run peak is at or below the operating limit. |
| `safety_label_agreement` | `1` if `source_one_step_safe` equals `long_run_safe` for the run; `safety_label_agreements` counts such runs. |
| `source_safe_to_long_unsafe` | `1` if the screen was `safe` but the paired long run was not (run or cell level). |
| `source_conservative_safe`, `long_conservative_safe` | Cell-level `safe` of the screens and of the long runs (all repetitions within margin). |
| `source_safe_explicit_phase_crossing` | `1` if the screen was `safe` but an explicitly marked phase of the long run crossed the operating limit. |
| `one_step_false_safe_rate` (estimand in `inference-summary.csv`) | Share of runs whose one-step screen was `safe` but whose longer run did not complete. |

In the margin-sensitivity tables (`threshold-cases.csv`, `threshold-transfers.csv`) each row states its own limit (`threshold_fraction`, `threshold_mib`; 90–100% of capacity), and every `safe` field of that row refers to that limit instead of 38,912 MiB; the primary limit is the 95% row.

These definitions apply to every table in the dataset and in the raw-evidence
archives, including the historical tables under `tables/` and
`electronics_revision_2026_10/`.
