# RLVRAMBench reproduction code

CPU-only reconstruction of the measurements, statistical tables, and four
figures supporting **RLVRAMBench: A Failure-Inclusive Benchmark for
GPU-Memory Feasibility in Colocated LoRA GRPO**.

Authors: Ondřej Kobza and Jan Šedivý, CIIRC, Czech Technical University in
Prague.

This repository separates the reproduction code from the larger research
workspace. The benchmark evidence is hosted at
<https://huggingface.co/datasets/kobzaond/RLVRAMBench>.
Both repositories are public as of 2026-09-14. Downloads require neither
authentication nor access approval. No DOI has been generated.

## What is reproduced

- 588 principal processes: 438 historical and 150 prospective revision runs.
- 16 additional historical source-screen processes, counted separately.
- All 40 published CSV/JSON result files, with only artifact-root path
  prefixes normalized for comparison.
- Four publication figures, regenerated as PDF and PNG.

The procedure verifies the archive and every evidence-manifest entry,
removes precomputed profiles from the analysis root, restores only the raw
paired-allocation provenance, and reconstructs the results from retained raw
executions and frozen matrices. Optional filesystem isolation hides both
the original project and reference tables during analysis.

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

Results appear in `reproduced/results/`, figures in `reproduced/figures/`,
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

The original scientific source snapshot is commit
`289747694f1a71052acc7977e742bfc0a45286a2` of
<https://github.com/kobzaond/rl>. The selected modules and frozen matrices are
copied unchanged. The download/extraction/comparison wrapper is new packaging
code and is tested separately. Some modules retain historical internal names;
these are implementation dependencies, not additional claims of this paper.

See `CITATION.cff` and `RIGHTS.md`. The original-material license choices,
final journal declarations, and DOI remain author-controlled. Public access
does not itself grant a software or data reuse license.
