# Supplementary query-level estimator comparison

This directory holds a **supplementary**, CPU-only comparison of memory
estimators on the colocated LoRA-GRPO targets of RLVRAMBench. It complements
the paper's native LLMem validation (Section "External estimator
validation"). That section runs LLMem's released estimator unmodified, on its
own training setting, and does not evaluate generation or the complete RL
loop.

Here, two kinds of estimator are applied directly to all 400 benchmark
queries and scored with the unchanged `benchmark.py evaluate`:

1. **DeepSpeed's published ZeRO model-state estimators** (ZeRO-2 and ZeRO-3),
   called for real, alone and in transparent composites.
2. A **transparent analytic per-stage budget** written for this stack, plus a
   variant with one calibrated constant.

The estimators read only the target settings, as exported by
`benchmark.py inputs`, plus architecture metadata. The calibrated variant
also reads the task's own *source* measurements. Target outcomes are read
only by the evaluator and for peak-error reporting, as `BENCHMARK.md`
requires. Nothing in `benchmark/` is modified, and nothing is retrained.

```bash
.venv-reproduce/bin/python estimator_comparison/run.py \
    --data benchmark --output estimator_comparison/results
.venv-reproduce/bin/python -m pytest -q tests/test_estimator_comparison.py
```

The run takes about 30 s and is deterministic: repeated runs are
byte-identical. `run.py` needs only the standard library and Matplotlib.
DeepSpeed is **not** imported there. Its recorded outputs are read from
`deepspeed_estimates.json`.

Units are MiB (2^20 B). The margin is 38,912 MiB and capacity is 40,960 MiB
per A100-40GB. A configuration is labelled `within_margin` if the predicted
maximum is ≤ margin, `memory_failure` if it exceeds capacity, and
`above_margin` otherwise.

## Files

| File | Content |
|---|---|
| `run.py` | Entry point |
| `estimator.py` | Analytic per-stage estimator, architecture loading, one-constant calibration |
| `estimator_eval.py` | Analytic predictions per task/track → `benchmark.py evaluate`; peak errors; figure |
| `deepspeed_collect.py` | Calls DeepSpeed's estimators (run in `.venv-deepspeed`), writes `deepspeed_estimates.json` |
| `deepspeed_estimates.json` | Version/source-hash record and all 180 DeepSpeed calls with returned bytes and printed tables |
| `deepspeed_baseline.py` | DeepSpeed-based predictors, scoring, peak errors; formula replica used only to verify the record |
| `configurations.py`, `common.py`, `plotting.py` | Shared loading of benchmark and panel configurations, I/O, figure style |
| `architecture_supplement.json` | Public `config.json` fields for Qwen2.5-1.5B and Mistral-7B-v0.3 (peak-error reporting only; neither is a query target) |
| `results/score_comparison_all_methods.csv` | Every method, every track, with reference rules |
| `results/{analytic,deepspeed}_peak_error_configurations.csv` | Predicted vs measured peaks, completed configurations |
| `results/analytic_failing_configurations_predictions.csv` | Analytic predictions for configurations whose repeated label is a memory failure |
| `results/{analytic,deepspeed}_{predictions,scores}/` | Per-track prediction CSVs and evaluator JSON |
| `results/figures/analytic_predicted_vs_measured.{pdf,png}` | Whole-run, generation and actor-stage predicted vs measured |

## DeepSpeed estimators: what was called

* **Versions.** DeepSpeed **0.19.8**, installed with `DS_BUILD_OPS=0`
  together with torch 2.14.1+cpu on Python 3.11, in the git-ignored
  `.venv-deepspeed`:

  ```bash
  python3.11 -m venv .venv-deepspeed
  .venv-deepspeed/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  DS_BUILD_OPS=0 .venv-deepspeed/bin/pip install deepspeed
  .venv-deepspeed/bin/python estimator_comparison/deepspeed_collect.py \
      --data benchmark --output estimator_comparison/deepspeed_estimates.json
  ```

* **Functions.**
  * `deepspeed/runtime/zero/stage3.py::estimate_zero3_model_states_mem_needs`
    (line 4016) and its reporter `estimate_zero3_model_states_mem_needs_all_cold`
    (line 4099); file sha256 `4ff3fdb4cbab…`.
  * `deepspeed/runtime/zero/stage_1_and_2.py::estimate_zero2_model_states_mem_needs`
    (line 4017) and `..._all_cold` (line 4073); file sha256 `752235af95d0…`.

  Full hashes and the printed tables are in the JSON. For example, Qwen2.5-3B
  on 2 GPUs prints "27.03GB | offload_param=none, offload_optimizer=none".
* **Check.** A formula replica in `deepspeed_baseline.py` must reproduce
  every recorded value exactly. It is checked at load and in the tests, and
  against live DeepSpeed when the test runs inside `.venv-deepspeed`.
* **Mapping.**
  * ZeRO-3 corresponds to FSDP full sharding, with num_gpus_per_node =
    gpu_count and zero_init = True.
  * `parameter_offload` maps to offload_param + offload_optimizer = cpu.
  * Parameter sets:
    * (a) `full`: all checkpoint parameters trainable, as the estimator
      assumes; the largest layer is max(transformer layer, embedding).
    * (b) `lora`: only the rank-64 all-linear adapters trainable, plus an
      explicitly added frozen bf16 base shard (2 B × params / GPUs, or one
      bf16 largest layer under parameter offload).
* **Composites**, all from target settings:
  * model states alone;
  * + vLLM reservation (u × 40,960);
  * + reservation + the analytic activation/logit term (below).

## Analytic per-stage budget

This is a static, first-principles budget in the style of
LLMem/DNNMem/ReaL. It is **not** an implementation of those tools. Each
stage is modelled per device:

* **Generation:** u × capacity, plus the resident FSDP weight shard (unless
  parameters are offloaded), plus LoRA params, grads and Adam states (rank
  64, fp32, sharded).
* **Actor update:** shard + LoRA states + a 2 × largest-unit bf16 all-gather
  + checkpointed activations + fp32 logits.
  * Activations are mb × (prompt + response) × hidden × layers × 2 B.
  * Logits are mb × seq × vocab × 4 B.
  * The reference policy is the adapter-disabled actor, so it adds no copy.
* **Log-prob:** the same terms, with the log-prob micro-batch.
* **Weight sync:** the full reservation (engine woken) + shard + one
  gathered tensor.
* **Retained state:** the reservation is added to the training stages when
  `free_cache_engine=False`.
* **Overhead:** a 1,024 MiB constant. The prediction is the maximum over
  stages.

Variants:

* `analytic_bf16`: the a-priori specification.
* `analytic_fp32_master`: fp32 FSDP master-weight shards. **Added after**
  observing that bf16 underestimated the generation stage in every completed
  historical configuration, so it is **not an independent test**.
* `calibrated_bf16`: the overhead is replaced by the mean residual over the
  task's completed *source* configurations. The fitted constants range from
  −15,847 to +6,710 MiB (median 3,189), itself a sign of stage
  misspecification.

## Results: all 400 queries

From `results/score_comparison_all_methods.csv`. Recall and precision
refer to within-margin approval. Failure and above-margin columns count
approved targets against their denominators.

| Method | Accuracy | Recall | Precision | Approved failures | Approved above-margin |
|---|---:|---:|---:|---:|---:|
| DeepSpeed ZeRO-3, full model states | 0.623 | 1.000 | 0.623 | 104/104 | 47/47 |
| ZeRO-3 full + reservation | 0.565 | 0.490 | 1.000 | 0/104 | 0/47 |
| ZeRO-3 full + reservation + activations | 0.563 | 0.486 | 1.000 | 0/104 | 0/47 |
| ZeRO-3 LoRA + frozen base | 0.623 | 1.000 | 0.623 | 104/104 | 47/47 |
| ZeRO-3 LoRA + frozen base + reservation | 0.623 | 1.000 | 0.655 | 84/104 | 47/47 |
| ZeRO-3 LoRA + frozen base + reservation + activations | 0.688 | 0.683 | 1.000 | 0/104 | 0/47 |
| DeepSpeed ZeRO-2, full model states | 0.518 | 0.743 | 0.680 | 62/104 | 25/47 |
| ZeRO-2 full + reservation | 0.310 | 0.080 | 1.000 | 0/104 | 0/47 |
| analytic_bf16 | 0.728 | 0.960 | 0.824 | 16/104 | 35/47 |
| analytic_fp32_master (post hoc) | 0.915 | 0.960 | 0.952 | 0/104 | 12/47 |
| calibrated_bf16 | 0.860 | 0.964 | 0.934 | 6/104 | 11/47 |
| source_label (reference) | 0.880 | 0.944 | 0.944 | 2/104 | 12/47 |
| always_approve / always_reject | 0.623 / 0.260 | 1 / 0 | 0.623 / – | 104 / 0 | 47 / 0 |

Per-track highlights (same file):

* **GPU-count transfer (36 queries).** ZeRO-3 LoRA + reservation is
  identical to always-approve: it approves 4/4 failing and 6/6 above-margin
  targets. analytic_bf16 approves 1 and 4, against source-copying's 2 and 4.
* **Model transfer (144 queries).** Approved failures are 0/40 for
  analytic_fp32_master, 1/40 for calibrated and 6/40 for analytic_bf16,
  against 0/40 for source-copying. Within-margin recall is 0.977 for all
  three analytic variants and 0.909 for source-copying.
* **Workload transfer (216 queries).** Source-label lookup is exact (1.000).
  No estimator matches it.
* **Horizon transfer (4 queries).** The calibrated variant and
  source-copying are both exact. The uncalibrated analytic variants score
  0.50.

## Results: peak error on completed configurations

The peak-error set is 251 configurations with completed repeated outcomes,
drawn from the benchmark and the later panels (replication control,
completion replication, admission). Error = predicted − measured whole-run
peak.

| Predictor | Mean signed | MAE | Failing configs (of 65) predicted within margin |
|---|---:|---:|---:|
| ZeRO-3 LoRA + frozen base | −31,210 (always under) | 31,210 | 65 |
| ZeRO-3 LoRA + frozen base + reservation | −5,932 (always under; −2,493 to −17,666) | 5,932 | 61 |
| ZeRO-3 LoRA + frozen base + reservation + activations | +9,753 | 11,568 | 7 |
| ZeRO-3 full | −11,636 (always under) | 11,636 | 46 |
| ZeRO-3 full + reservation | +13,642 | 15,974 | 6 |
| analytic_bf16 | −1,769 (81% under) | 4,648 | 32 |
| analytic_fp32_master (post hoc) | +1,087 (median AE 1,091) | 3,058 | 10 |

Stage-level errors for analytic_bf16:

* **Generation:** underestimated in all 232 configurations with stage
  data (mean −3,814 MiB).
* **Actor update:** overestimated (mean +4,524 MiB), because worst-case
  max-length logits are not what VERL materialises.
* **7B models:** underestimated by about 8 GB (Qwen −8,073, Mistral −8,579).

The fp32 variant's 10 failing configurations predicted within margin are:

* six offloaded 7B runs on 2 GPUs, which fail by weight-sync OOM;
* four 7B/Mistral 4-GPU runs at u = 0.7, which fail at vLLM start-up
  admission.

## Interpretation

Training-state estimators of the DeepSpeed, LLMem and DNNMem class model
the weights, gradients, optimizer states and, for some, activations of a
training step. They have **no term for a colocated inference engine**.

* **The reservation is the omitted term.**
  * In the median completed configuration, DeepSpeed's LoRA + frozen-base
    model states are 3,983 MiB, 10.5% of the 35,958 MiB measured peak. The
    vLLM reservation is 24,576 MiB, 73.8%.
  * Used as intended, with a correctly specified LoRA trainable set, the
    estimator approves every benchmark target, including all 104
    memory-failing ones.
  * Under its own full-trainable assumption (median 27,674 MiB) it still
    underestimates every completed peak.
* **The reservation is necessary but not sufficient.** LoRA model states +
  reservation still underestimate all 251 peaks and approve 84/104 failing
  targets. They omit:
  * the colocated non-reservation residency;
  * the weight-sync transient;
  * vLLM's start-up admission check.
* **Additive composites over-count.** Summing training activations onto the
  generation reservation adds stages that do not coexist when state is
  released. The result is conservative (0 approved failures) but rejects
  79/249 usable targets.
* **Retained state is a separate omitted term.** Without state release,
  Qwen2.5-1.5B's measured actor stage is 40,458 MiB against 2,116 MiB of
  DeepSpeed LoRA model states. The gap is largely the retained 28,672 MiB
  reservation.
* **Stage-wise modelling is needed.** An estimator for this setting has to
  model stage-wise maxima (reservation + residency during generation and
  sync; model states + activations + retained state during the update),
  not a single training sum. Even then, the transparent analytic budget
  makes errors of 1–4 GiB, which is comparable to the 2 GiB margin band.

## Caveats

* One stack (VERL + vLLM, LoRA-GRPO, A100-40GB), models ≤ 7.6 B, and 94
  distinct target configurations reused across 400 queries. These are not
  400 independent experiments.
* The DeepSpeed estimators are used outside their intended scope, which is
  ZeRO training without an inference engine. The mapping to FSDP/LoRA is
  documented above. The poor scores reflect the missing colocated terms and
  are not a defect of DeepSpeed for its own setting.
* The analytic stage structure and the fp32 variant were chosen with the
  historical stage traces visible. Only the later prospective panels are
  out-of-design checks, and the largest errors appear there (7B models).
* Failed processes have no completed peak. Peak errors use completed
  configurations only, and failure handling is reported through labels.
* Qwen2.5-1.5B and Mistral-7B architecture fields come from public
  `config.json` files and are used only in peak-error tables.
