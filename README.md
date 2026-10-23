# RLVRAMBench: evaluate resource decisions and reproduce the evidence

CPU-only reconstruction of the measurements, tables, and figures supporting
**Memory Feasibility in Colocated Language-Model Reinforcement Learning:
A Failure-Aware Measurement Study**. RLVRAMBench is the accompanying evidence
and configuration-evaluation artifact for the studied LoRA-GRPO execution
stack on A100 hardware, with models up to approximately 7.6 billion parameters,
plus a cross-accelerator replication on AMD MI250X. It is not a general
benchmark of reinforcement-learning systems.

Authors: Ondřej Kobza and Jan Šedivý, Czech Institute of Informatics,
Robotics and Cybernetics, Czech Technical University in Prague.

This repository separates the reproduction code from the larger research
workspace. The benchmark evidence is hosted at
<https://huggingface.co/datasets/kobzaond/RLVRAMBench>.
Both repositories are public as of 2026-09-14. Downloads require neither
authentication nor access approval. The dataset DOI is `10.57967/hf/10780`.

**Electronics submission version:** tag `electronics-submission-2026-10-23`
in this repository and on the Hugging Face dataset. Hugging Face `main`
contains this version, including the paper, all four raw-evidence archives,
curated results, and validation reports. See
[the submission release guide](docs/SUBMISSION_RELEASE.md) for downloads,
archive coverage, and the complete CPU reconstruction commands.

## Start with the benchmark

Predict whether a target configuration completes within the device-memory
margin, completes above it, or has a diagnosed memory failure. The curated
data directory supplies explicit settings, repeated outcomes, linked attempts,
and four transfer tracks. It contains 52 tasks with 400 queries reusing
94 distinct target configurations, not 400 independent experiments.

The lightweight evaluator needs only Python 3.11 or newer:

```bash
python3 benchmark.py verify
python3 benchmark.py baseline --output predictions.csv
python3 benchmark.py evaluate --predictions predictions.csv --output scores.json
```

For your method, export permitted inputs with `benchmark.py inputs`,
then provide a prediction CSV. Read `BENCHMARK.md` for the information
boundary and `DATA_DICTIONARY.md` for all fields and units. Target outcomes
are public for inspection, but are not permitted prediction inputs.
Within-model workload transfer is already solved by lookup on this grid;
this release is not a hidden leaderboard for sophisticated predictors.

For a resource decision, read within-margin recall together with approval
precision and separate approvals of memory failures and above-margin
completions. Pooled label accuracy is not an adequate deployment ranking:
on GPU-count transfer, always approving has higher accuracy than copying
the source labels but approves more memory failures. Transparent reference
scores are in `benchmark/reference_scores.json`.

The separate admission panel measures decisions as target tests become
available. It fixes four model–workload cases and twelve candidates, with
one screening seed and three **different** evaluation seeds per candidate.
A deterministic replay requests screens under a budget; it never uses
those screens as their own repeated evaluation outcome.

```bash
python3 decision_benchmark.py replay \
  --protocol benchmark/decision/protocol.json \
  --attempts benchmark/decision/results/attempts.json \
  --output admission-scores.json
```

Compare usable configurations recovered at each target-attempt budget,
keeping failure/margin approvals and donor investment visible. Twelve
screening runs and 36 hidden evaluation runs form the physical 48-slot
panel; they are not 48 independent cases or 48 free policy observations.

The following sections describe the separate, heavier reconstruction of
the paper's results from raw evidence.

## Optional estimation extension

Releases with `expected_estimation_derived_files` in `release.json` also
include a separate estimation comparison. Fixed references copy donor labels,
fit empirical component regression with a startup-budget check, or fit
multinomial logistic classification. Eight architecture/configuration size
proxies are not an exact tensor-liveness model. Three retrospective folds
withhold one model family each; historical outcomes were visible during
method design.

The prospective schedule has twelve configurations and 36 planned slots,
not 36 guaranteed eligible outcomes. The release's `estimation_counts`
come from the collector and retain unresolved processes and configurations.
The original protocol, amendment, frozen predictions, complete expected execution-commit
map, and any scheduler accounting remain separate from derived outcomes.
Earlier releases without this extension follow the unchanged reconstruction
path.

For a release containing the frozen estimator inputs, replay and compare
fresh fits without reading prospective outcomes:

```bash
.venv-reproduce/bin/python -m memory_tuner.verify_estimation \
  --root . --refit --output estimator-verification.json
```

The output must not already exist. Numerical tolerances are reported and
categorical decisions must match exactly. Recorded backend-discovery warnings
limit thread-count and CPU-performance claims; passing numerical checks does
not establish that every numerical library used one thread.

### Separate expanded-host follow-up

Releases with `expected_host_capacity_derived_files` also reconstruct
`benchmark/host_capacity/`. This follow-up keeps the original predictions
and uses new seed slots under an expanded host-resource condition. It does
not replace the original larger-model panel or refit its predictors.

All eighteen invocations launched: nine two-GPU startup memory-admission
failures are eligible, while nine four-GPU invocations lack the required
final-step record and remain unresolved. Returning update/synchronization
calls and a zero exit are not substituted for that missing evidence.
Only three of six repeated configuration labels resolve, all to failure.
No completed-peak error or recovery of a usable setting is established.

The wrapper verifies both frozen designs, hides both target-outcome trees
from the historical refit, checks prediction inheritance without a host
refit, and independently reconstructs the two thirteen-file result sets.
Coverage and costs stay separate. See the cohort guides and `release.json`
for exact versions and observed counts.

## Separate replication panels

Releases with `replication_reproduction` in `release.json` add two panels
alongside the earlier estimation and expanded-host cohorts.

The larger-model completion replication compares two- and four-GPU runs of
Qwen2.5-7B and Mistral-7B-Instruct-v0.3. Its 108 slots yield 35 validated
completions, 72 GPU-memory failures, and one original infrastructure failure
retained as invalid without retry. Of 36 configuration labels, 35 resolve:
seven within-margin, four above-margin, and 24 memory
failures. The frozen logistic classifier approves six within-margin
configurations, no resolved memory-failing configuration, and one resolved
above-margin configuration. It also approves the unresolved configuration,
whose eligible completions already exceed the memory limit. All six useful
classifier approvals use four GPUs at c1, with actor parameter offload enabled.
These are one-step measurements without validation or checkpoints, not
evidence of full-training feasibility.

The batch/reservation replication control specifies 36 blocks, 144 slots,
and twelve model–workload cases, with full instrumentation fixed.
All slots launched: 140 completed with validated evidence and the
original four remain unresolved. Thirty-five complete blocks cover
all twelve cases, with two blocks for Qwen on standard code and three
elsewhere. The actor-stage batch contrast is positive in every complete
block; its equal-case mean is 3,904.61 MiB, compared with 7.46 MiB in
the whole-run batch contrast. The whole-run reservation contrast averages
4,060.26 MiB. Full instrumentation remains fixed, so these results do
not establish logging independence.

The wrapper verifies the archived exact source snapshots, freezes, and
runtime amendments, then reconstructs four scientific outputs per panel
from raw evidence. It separately replays the frozen completion predictions
using the original 90-configuration fit, without refitting or reading target
outcomes. The eight scientific outputs are:

```text
reproduced/completion_replication-results/
  summary.json
  slots.csv
  configurations.csv
  pairs.csv
reproduced/replication_control-results/
  replication_control_summary.json
  replication_control_trials.csv
  replication_control_blocks.csv
  replication_control_cases.csv
```

Comparison normalizes the documented artifact and source path prefixes.
Only `collection_cpu_wall_seconds` in the completion `summary.json` is
excluded from content equivalence; that summary is compared as canonical
JSON, and both CPU collection times remain in `verification.json`.
The other seven files must match exactly after path normalization, with
no field exclusions for the control.

## Independent offload factorial

`benchmark/offload_factorial/` holds the later experiment that varies actor
parameter offload, actor micro-batch (4/8), and generation reservation
(0.45/0.70) independently on four GPUs for Qwen2.5-7B and
Mistral-7B-Instruct-v0.3 on three workloads. `freeze.json` seals
`protocol.json`, `matrix.csv`, and the six methods' approval decisions in
`predictions.json` before execution (source commit `3d6ecac`). All 144 slots
are eligible without retry: 108 complete within margin and 36 fail the
generation engine's startup budget check. The 48 configurations each have
three agreeing seeds: 36 within margin and 12 memory failures (every
resident-actor configuration at reservation 0.70).

| File in `results/` | Unit |
|---|---|
| `slots.csv` | One of 144 invocations, eligibility, peaks per stage, diagnostics |
| `configurations.csv` | One of 48 settings, three-seed label, each frozen approval |
| `prediction_scores.csv` | Approvals/rejections per method, overall and by stratum |
| `offload_pairs.csv` | One of 72 matched offload-on/off comparisons |
| `block_contrasts.csv` | Within-block factor contrasts where measurements exist |
| `summary.json` | Coverage, transitions, case contributions, accounting, raw-evidence hashes |

`panel_group=anchors` marks the six original-c1 settings; the other 42 are
`challenge`. A missing continuous contrast is not zero: completed-peak
differences need both runs to complete, so the 0.70 stratum has none.
`allocation_accounting.json` holds terminal scheduler records; its
whole-allocation cost is not added to invocation costs. Data-dictionary
entries for this panel are kept here, not in `DATA_DICTIONARY.md` or
`BENCHMARK.md`, because those two files are hash-bound to the pinned
release in `source-provenance.json` and `reproduce.py` rejects any change.

The CPU check `tests/test_offload_factorial_paper_numbers.py` recomputes the
manuscript's numbers from the curated files. It reads the labels from
`slots.csv` and the approvals from the sealed `predictions.json`, and uses
the derived tables only as cross-checks. It covers the outcome table, the
prediction table (combined rule 36/0/0, frozen logistic 29/0/7, component
regression 36/6/0, offload only 24/0/12, reservation only 24/0/12, c1 only
6/0/30), and the matched reductions at reservation 0.45 (equal-case means
3,226 and 3,223 MiB, range 1,868–4,296 MiB, weight sync setting every
offloaded peak). It also verifies the hashes in `freeze.json` and in
`benchmark/revision_20260917/offload-independent-reconstruction.json`.

That report records the authors' relocated raw-evidence reconstruction
(50,058 evidence files, five CSVs byte-identical, summary identical apart
from `raw_root`/`source_snapshot`). Repeating that reconstruction requires
the raw per-block evidence under `output/offload_factorial/`, which is not
part of the pinned Sept-17 archive. The submission dataset now supplies
that evidence separately as `releases/RLVRAMBench-offload-raw-20261022.tar.zst`,
including the frozen source and accounting. The submission release guide
shows how to re-derive `slots.csv` and the other five outputs. The collection and
analysis modules (`memory_tuner/collect_offload_factorial.py`,
`offload_factorial_analysis.py`, `offload_factorial_protocol.py`,
`offload_factorial_runtime.py`) and their unit tests are included for that
purpose. `run_offload_factorial.py/.slurm`, `watch_offload_factorial.py`, and
`snapshot_offload_factorial.py` are launch-only provenance. Their hashes are
listed under `offload_factorial_extension` in `source-provenance.json`,
separately from the pinned release closure.

## Electronics revision (2026-10)

The MDPI *Electronics* revision adds four frozen A100 RL panels and a native
LLMem comparison.

**Curated results.** The results are in `profiles/electronics_revision/`, and the
frozen protocols, matrices, and resource accounting are in
`benchmark/electronics_revision_20261016/`. The panels are:

| Panel | Runs | Within | Above | Memory failure | Unresolved |
|---|---:|---:|---:|---:|---:|
| Full-parameter / LoRA and no-reference controls | 108 | 106 | 0 | 0 | 2 |
| Reservation 0.45–0.70 (7B) | 144 | 111 | 11 | 21 | 1 |
| Generation-state release sweep | 36 | 17 | 6 | 12 | 1 |
| Twenty-update operating cost | 24 | 24 | 0 | 0 | 0 |

The native LLMem comparison has 24 prediction/measurement pairs, with an
equal-case MAE of 236.6 MiB and a mean absolute relative error of 5.37%.

**Source and layout versions.** The scientific analysis modules originate
from workspace branch `electronics-revision-20261016` (commit `4febe59`).
The 23 October submission updates only the figure and table renderers in
`memory_tuner/electronics_*` and `amd_mi250x/make_supplement.py`.
`electronics_historical_figures.py` supplies the revised boundary and logging
plots; the hash-bound historical `make_review_figures.py` remains intact.
`collect_llmem_native.py` and `llmem_native*.py` retain their recorded versions. The unmodified upstream estimator is in
`third_party/llmem_estimator/` with its provenance note.
`docs/REPRODUCING_ELECTRONICS_REVISION.md` and
`docs/electronics_measurement_audit.md` are the authors' reconstruction guide
and telemetry audit.

**Numbers check.** `tests/test_electronics_revision_paper_numbers.py` checks
the manuscript's revision numbers against the curated files.

**Not portable.** Launch-design tests that are hard-wired to the Karolina
cluster paths are not included.

**Raw archive.** Full raw reconstruction uses the separate evidence archive
`electronics-full-20261021-a53af63.tar.zst`
(SHA-256 `f70f639fc0c991e79b9e140d4c26482a9c3490000abe3c5acf6bc62fb7a599bf`).
It is not part of this repository, and its historical part also needs
`profiles/standard_grpo/` tables from that archive. The submission dataset
publishes the complete archive in `releases/`; no author contact is needed.

## Cross-accelerator replication (AMD MI250X)

`amd_mi250x/` replays the A100 configurations on the LUMI supercomputer. It
uses MI250X GCDs (65,520 MiB) with ROCm 7.2.3, vLLM 0.25.1 and VERL 0.9.0. It
changes hardware and software together, so it is a transfer test rather than
a hardware ablation.

The panel has 462 planned one-step and 40-step invocations in five parts:
- the A100 GSM8K/MATH grid
- the 7B models
- a reservation sweep from 0.30 to 0.95
- GRPO-LoRA, PPO with a full-parameter critic, and full-parameter GRPO
- one-step versus 40-step runs

**Results.** Of the 90 replayed configurations, 89 obtain a complete
three-repetition label and 79 of these keep their A100 label; one is
unresolved (two memory failures and one non-memory failure). Two findings differ from A100:
- **Failure stage.** Infeasible settings fail after the actor update, while
  vLLM restores its released state, instead of at vLLM startup.
- **Whole-run peak.** The whole-run peak no longer hides actor-stage growth.

**Files.**
- `amd_mi250x/README.md`: harness, outcome rules and the four runtime amendments.
- `amd_mi250x/results/`: per-trial and per-configuration outcomes.
- `amd_mi250x/results/analysis/`: the comparison with the A100 tables.

**Reproduce the tables on a CPU.** The tables are reproducible from those
outputs:

```bash
python3.11 amd_mi250x/analyze.py
python3.11 amd_mi250x/make_supplement.py --output s21.tex
```

`collect.py` rebuilds `results/trials.csv`, `configurations.csv` and
`summary.json` byte for byte from the raw evidence archive
`data/mi250x/RLVRAMBench-mi250x-raw-20261022.tar.zst` in this repository
(SHA-256 `400f75b1…67ae`; see
`amd_mi250x/results/raw_reconstruction_verification.json`):

```bash
mkdir mi250x-raw && tar --use-compress-program=unzstd \
  -xf data/mi250x/RLVRAMBench-mi250x-raw-20261022.tar.zst -C mi250x-raw
(cd mi250x-raw && sha256sum -c --quiet release/MANIFEST.sha256)
python3.11 amd_mi250x/collect.py --root mi250x-raw --output mi250x-raw/results
```

A completion needs the final requested step and exit code 0. Every
eligible outcome needs the log, the device map, and non-empty memory and
stage traces with one column per task GPU. A repeated label needs three
eligible repetitions. `results/attempt_ledger.csv` lists every Slurm
execution, including the one restart after a node failure.

## Estimator comparison

`estimator_comparison/` applies two estimators to all 400 benchmark targets and
scores them with `benchmark.py evaluate`:
- DeepSpeed's published ZeRO-2/ZeRO-3 model-state estimators, called from
  DeepSpeed 0.19.8 and recorded in `deepspeed_estimates.json`
- a transparent analytic component budget

**Results.** Model states alone approve every memory-failing target, and
adding the vLLM reservation still approves 84 of 104. The comparison
complements the paper's native LLMem validation. It does not test DeepSpeed
within its intended training scope. See `estimator_comparison/README.md`.

## What is reproduced

- 588 principal processes: 438 historical and 150 prospective revision runs.
- 16 additional historical source-screen processes, counted separately.
- All 40 published CSV/JSON result files, with only artifact-root path
  prefixes normalized for comparison.
- 27 additional audit/control CSV/JSON files, including all 657 recorded
  attempts underlying the original 588 retained outcomes.
- The review-requested control: 26 started processes, 24 within-margin
  completions, and six usable four-condition blocks. Initial noncompletions
  remain distinct from the two additional allocations.
- Four publication figures, regenerated as PDF and PNG.
- The curated benchmark tables, task definitions, baseline predictions,
  and scores, compared byte-for-byte after regeneration.
- Eleven admission-panel outputs, including every recorded attempt,
  independent repeated labels, per-case acquisition transcripts, and
  three admission rules at four measurement budgets. Reference target
  outcomes are hidden from their own reconstruction.
- When present, the estimation collector's exact output file set, independent
  raw reconstruction, frozen prediction replay, and a separately reported
  fresh-fit comparison. The expected file count and observed coverage are
  taken from `release.json`, not assumed from the planned schedule.
- When present, the separate host-capacity collector's thirteen outputs,
  its inherited-prediction check, and independent raw reconstruction under
  the original completion criteria.
- When present, the eight replication-panel scientific outputs and frozen
  completion prediction replay described above.

The procedure verifies the archive and every evidence-manifest entry,
removes precomputed profiles from the analysis root, restores only the raw
paired-allocation provenance, and reconstructs the results from retained raw
executions and frozen matrices. With `--isolate-analysis`, filesystem isolation
hides the original project, reference tables, and the code checkout's
precomputed curated views during analysis. Each child checks that the hidden directories
are inaccessible as evidence. Use repeatable `--hide-path DIRECTORY`
arguments for other known copies of the project or reference evidence;
the reconstruction output must be outside those directories.

This is reproduction of completed measurements, not a rerun of GPU training.
It does not require model weights, CUDA, an A100 allocation, or Slurm.
It does not establish learning-quality improvements or generalization beyond
the reported configurations in this single LoRA-GRPO/A100 software stack.

Some historical source records identify modified working trees without
preserving their per-run changes as patches. The saved measurements can
be reanalyzed, but a commit ID alone does not recover those modifications
for an exact GPU rerun. Later controls and the admission panel use
immutable execution sources.

## Requirements

Use CPython 3.11 on Linux, `zstd`, and the pinned publication dependencies.
For this revised full release, `bwrap` (bubblewrap) must be available and
permitted to create the isolated child processes required by
`--isolate-analysis`. Allow sufficient free disk space for archive extraction,
separate reference copies, and reconstructed outputs; the required space
depends on the release.

```bash
bash memory_tuner/bootstrap_publication.sh
```

This creates `.venv-reproduce` using `publication-requirements.txt`.
Run environment-creation commands from the project directory; keep every
virtual environment inside the project, not in your home directory.
Use the complete staged requirements, including the estimation dependencies
when that extension is present. The copied analysis
modules and matrices are recorded, with their SHA-256 digests, in
`source-provenance.json`. Estimation reconstruction checks those standalone
files against both this record and their archived source copies before
executing analysis.

## Get the frozen benchmark

The archive filename and SHA-256 are pinned in `release.json`. A published
release also pins the dataset commit in `huggingface_revision`. If that field
is `null`, the release is prepared locally but has not yet been published:
use the accompanying verified archive, not the download helper or an older
public archive. After publication, you can download the pinned archive
manually from the Hugging Face dataset repository.

For the download helper, use a separate environment:

```bash
python3.11 -m venv .venv-download
.venv-download/bin/python -m pip install -r download-requirements.txt
.venv-download/bin/python download.py --output-dir downloads
```

No Hugging Face account or token is required for this public benchmark.
Credentials must not be placed in source files, Git URLs, or `release.json`.

## Reproduce

Replace `ARCHIVE.tar.zst` below with the downloaded filename printed by
`download.py`. The output directory must not already exist.

```bash
.venv-reproduce/bin/python reproduce.py \
  --archive downloads/ARCHIVE.tar.zst \
  --work-dir reproduced \
  --isolate-analysis
```

Releases containing either replication panel require `--isolate-analysis`;
this revised full release has no non-isolated fallback. Only for historical
releases without replication panels may you omit the flag if bubblewrap is
unavailable. Those runs still perform raw reconstruction and numerical
comparisons, but do not claim filesystem isolation.

Original results appear in `reproduced/results/`, revised audits and controls
in `reproduced/review-results/`, figures in `reproduced/figures/`,
curated data in `reproduced/benchmark/`,
admission results in `reproduced/decision-results/`,
and the machine-readable acceptance report in `reproduced/verification.json`.
The reference profiles are retained separately under
`reproduced/reference-profiles/`; they are never placed back into the
analysis input tree, apart from the required raw paired-allocation records.

For an estimation release, `reproduced/estimation-verification.json` records
replay and fresh-fit checks, and `reproduced/estimation-results/` contains
the independently collected outputs. Only the frozen protocol, amendment,
matrix, architecture metadata, target settings, fitted models, predictions,
seal, execution-commit map, and used allocation accounting are restored from
the estimation reference directory. Its `results/` directory is never restored.
Historical fitting tables come from the preceding regeneration.

With `--isolate-analysis`, all analysis children are denied the reference
directories. The replay/refit child is additionally denied prospective raw
records, scheduler logs, and manuscript files containing reported outcomes;
the collector can read the raw inputs afterward.
Reference comparison happens in the parent process. Estimation CSV/JSON
contents must match after artifact-root path normalization; each output
manifest is independently verified against its own files, since relocated
paths can change byte hashes. Coverage counts, the prediction-seal hash, and
the execution-commit map must also match the release.

The final report separates `estimation_independent_raw_reconstruction_passed`,
`estimation_prediction_replay_passed`, and `estimation_fresh_refit_compared`.
The two estimation isolation flags are false when filesystem isolation is
omitted; that run cannot satisfy the isolated publication gate.

## Tests

```bash
.venv-reproduce/bin/python -m pytest -q
```

Tests cover path confinement, archive link validation, statistical utilities,
raw lifecycle checks, prospective-matrix invariants, independent evaluation
seeds, acquisition budgets, and unresolved-outcome accounting. The full archive
reconstruction above is a separate integration check.

## Source and citation

The current scientific source snapshot of <https://github.com/kobzaond/rl>
is pinned in `source-provenance.json`. The selected modules and frozen
matrices are copied unchanged. The download/extraction/comparison wrapper is packaging
code and is tested separately. Some modules retain historical internal names;
these are implementation dependencies, not additional claims of this paper.

The earlier batch-by-logging control supports positive observed actor-stage
batch contrasts without allocator logging. Generated responses vary between conditions,
so the measurements do not establish pure tracing overhead on identical
tensors, logging equivalence, or a universal correction factor. Earlier
immutable evidence releases remain available in the dataset repository.

Original benchmark data and reproduction code are MIT licensed; see
`LICENSE`, `RIGHTS.md`, and `CITATION.cff`. Third-party terms remain separate.
Final journal declarations, submission approval, and DOI creation remain
author actions.
