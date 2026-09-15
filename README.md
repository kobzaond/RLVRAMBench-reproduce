# RLVRAMBench: evaluate resource decisions and reproduce the evidence

CPU-only reconstruction of the measurements, tables, and figures supporting
**Memory Feasibility in Colocated Language-Model Reinforcement Learning:
A Failure-Aware Measurement Study**. RLVRAMBench is the accompanying evidence
and configuration-evaluation artifact for the studied LoRA-GRPO execution
stack on A100 hardware, not a general benchmark of reinforcement-learning systems.

Authors: Ondřej Kobza and Jan Šedivý, Czech Institute of Informatics,
Robotics and Cybernetics, Czech Technical University in Prague.

This repository separates the reproduction code from the larger research
workspace. The benchmark evidence is hosted at
<https://huggingface.co/datasets/kobzaond/RLVRAMBench>.
Both repositories are public as of 2026-09-14. Downloads require neither
authentication nor access approval. No DOI has been generated.

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

The procedure verifies the archive and every evidence-manifest entry,
removes precomputed profiles from the analysis root, restores only the raw
paired-allocation provenance, and reconstructs the results from retained raw
executions and frozen matrices. Optional filesystem isolation hides
the original project, reference tables, and the code checkout's precomputed
curated views during analysis. Each child checks that the hidden directories
are inaccessible as evidence. Use repeatable `--hide-path DIRECTORY`
arguments for other known copies of the project or reference evidence;
the reconstruction output must be outside those directories.

This is reproduction of completed measurements, not a rerun of GPU training.
It does not require model weights, CUDA, an A100 allocation, or Slurm.
It does not establish learning-quality improvements or generalization beyond
the reported compact-LoRA/A100 configurations.

Some historical source records identify modified working trees without
preserving their per-run changes as patches. The saved measurements can
be reanalyzed, but a commit ID alone does not recover those modifications
for an exact GPU rerun. Later controls and the admission panel use
immutable execution sources.

## Requirements

Use CPython 3.11 on Linux, `zstd`, and the pinned publication dependencies.
Allow roughly 4 GB of free disk space for extraction, reference copies, and
outputs. `bwrap` (bubblewrap) is optional for filesystem-isolation checking.

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

The exact dataset commit, archive filename, and SHA-256 are pinned in
`release.json`. You can download that archive manually from the Hugging Face
dataset repository. Do not substitute an older review archive.

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

Omit `--isolate-analysis` if bubblewrap is unavailable. The same raw
reconstruction and numerical comparisons are still performed, but filesystem
isolation is then explicitly not claimed.

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

The new control supports positive observed actor-stage batch contrasts
without allocator logging. Generated responses vary between conditions,
so the measurements do not establish pure tracing overhead on identical
tensors, logging equivalence, or a universal correction factor. Earlier
immutable evidence releases remain available in the dataset repository.

Original benchmark data and reproduction code are MIT licensed; see
`LICENSE`, `RIGHTS.md`, and `CITATION.cff`. Third-party terms remain separate.
Final journal declarations, submission approval, and DOI creation remain
author actions.
