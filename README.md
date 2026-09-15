# RLVRAMBench: evaluate resource decisions and reproduce the evidence

CPU-only reconstruction of the measurements, statistical tables, and four
figures supporting **RLVRAMBench: A Benchmark for Memory Feasibility in
Colocated Language-Model Reinforcement Learning**.

Authors: Ondřej Kobza and Jan Šedivý, CIIRC, Czech Technical University in
Prague.

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

The following sections describe the separate, heavier reconstruction of
the paper's results from raw evidence.

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

The procedure verifies the archive and every evidence-manifest entry,
removes precomputed profiles from the analysis root, restores only the raw
paired-allocation provenance, and reconstructs the results from retained raw
executions and frozen matrices. Optional filesystem isolation hides both
the original project, reference tables, and precomputed curated views during analysis.

This is reproduction of completed measurements, not a rerun of GPU training.
It does not require model weights, CUDA, an A100 allocation, or Slurm.
It does not establish learning-quality improvements or generalization beyond
the reported compact-LoRA/A100 configurations.

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
The scientific dependency versions are unchanged from the original paper
release. The copied analysis modules and matrices are recorded, with their
SHA-256 digests, in `source-provenance.json`.

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
and the machine-readable acceptance report in `reproduced/verification.json`.
The reference profiles are retained separately under
`reproduced/reference-profiles/`; they are never placed back into the
analysis input tree, apart from the required raw paired-allocation records.

## Tests

```bash
.venv-reproduce/bin/python -m pytest -q
```

Tests cover path confinement, archive link validation, statistical utilities,
raw lifecycle checks, and prospective-matrix invariants. The full archive
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
