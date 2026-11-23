# RLVRAMBench: reproduction code

CPU-only reconstruction of every measurement, table, and figure reported in
**Memory Feasibility in Colocated Language-Model Reinforcement Learning: A
Failure-Aware Measurement Study**, from the released raw evidence. RLVRAMBench
measures GPU-memory feasibility when language-model generation and LoRA-GRPO
updates share A100 GPUs (VERL/vLLM; models up to about 7.6 billion
parameters), with a cross-accelerator replication on AMD MI250X. It is not a
general benchmark of reinforcement-learning systems.

Author: Ondrej Kobza ([ORCID 0000-0002-0529-9860](https://orcid.org/0000-0002-0529-9860)),
Czech Institute of Informatics, Robotics and Cybernetics, Czech Technical
University in Prague.

## Release `rlvrambench-2026-11`

One version identifier is used by the paper, the dataset, this repository,
and `release.json`:

| Item | Identifier |
|---|---|
| Version tag (dataset and code) | `rlvrambench-2026-11` |
| Dataset | <https://huggingface.co/datasets/kobzaond/RLVRAMBench>, commit `DATASET-COMMIT-TO-BE-SET` |
| Dataset DOI | `10.57967/hf/TO-BE-MINTED` (the earlier DOI 10.57967/hf/10780 belongs to an older revision) |
| Code | this repository, the commit tagged `rlvrambench-2026-11` |
| Manifest | `release.json` (archives, hashes, procedures, environments) |
| Environments | CPython 3.11 + `publication-requirements.txt`; CPython 3.12 + `revision_2026_11/cpu_reanalyses/requirements.txt` |

Raw-evidence archives (directory `releases/` of the dataset):

| Archive | Bytes | SHA-256 | Content |
|---|---:|---|---|
| `RLVRAMBench-review-artifact-submission-20260917-b0018d0.tar.zst` | 210,126,067 | `3dd90eaca2f2d9b5eceb955fe67b6a3528210a6bb9bd7684ac3eb3c36d84a5ab` | historical A100 studies, admission, estimation, replications |
| `RLVRAMBench-offload-raw-20261022.tar.zst` | 33,356,651 | `ac2c69bae8d7e0af07e27de9c182e095fa77a28f5b0d0fc906fbfff302040041` | independent-control (offload) experiment |
| `electronics-full-20261021-a53af63.tar.zst` | 74,609,840 | `f70f639fc0c991e79b9e140d4c26482a9c3490000abe3c5acf6bc62fb7a599bf` | A100 follow-ups (adaptation, reservation, release, timing) and native LLMem |
| `RLVRAMBench-mi250x-raw-20261022.tar.zst` | 58,527,374 | `400f75b1c3015d9903b3e898b0ed2dc3771b07363bff7f8761aec0c798ed67ae` | MI250X replication |
| `RLVRAMBench-revision-raw-20261123.tar.zst` | 49,842,735 | `87dada01f6f7d25e01210bbfc24b7faf35041e96e3731e8bce9e24766106fe5d` | revision experiments E1–E3 (November 2026) |

## Fast path: reproduce everything

Requirements: Linux, CPython 3.11 and 3.12, `tar`, `zstd`, and bubblewrap
(`bwrap`, allowed to create unprivileged namespaces); about 10 GB of free
disk space. No GPU, Slurm, model weights, or credentials.

```bash
git clone https://github.com/kobzaond/RLVRAMBench-reproduce.git
cd RLVRAMBench-reproduce
git checkout rlvrambench-2026-11
bash reproduce_all.sh --revision DATASET-COMMIT-TO-BE-SET --work-dir reproduce-all-output
```

`reproduce_all.sh` creates `.venv-reproduce` (3.11) and `.venv-reanalysis`
(3.12) inside the checkout (override the interpreters with
`RLVRAM_BOOTSTRAP_PYTHON` and `RLVRAM_REANALYSIS_PYTHON`), downloads the five
archives at that dataset commit over HTTPS, checks size and SHA-256, and runs:

1. `historical`: `reproduce.py --isolate-analysis` on the historical archive
   (40 original and 27 audit/control files, curated benchmark, admission,
   estimation, expanded-host and replication panels, four figures).
2. `offload`: `memory_tuner.collect_offload_factorial` (five tables byte
   for byte).
3. `a100`: the archive's own verifier (46 outputs), then the same analysis
   modules into a persistent directory with saved results unreadable, and the
   figure and SI-table renderers.
4. `mi250x`: `amd_mi250x/collect.py`, `analyze.py`, `make_supplement.py`.
5. `estimator`: `estimator_comparison/run.py` and `make_supplement.py`.
6. `revision`: the E1–E3 and smoke-gate collectors from the frozen source
   snapshots inside the revision archive, the analyses, and
   `analysis.py check-existing` (earlier reported values).
7. `cpu_reanalyses`: retrospective analyses C1, C2/C3/C8, C4–C6, compared
   with `revision_2026_11/cpu_reanalyses/*/reference/`; plus an informational
   rerun under the 3.11 environment.
8. `figures`: the paper figures under their manuscript file names
   (`fig1_reservation_sensitivity.pdf` … `fig7_horizon_agreement.pdf` and SI
   Figures S1–S3), compared byte for byte with the manuscript files.
9. `tests`: `pytest -q tests`.

The work directory then holds `REPORT.md` and `report.json`: every step with
its commands, checks, duration and outputs, the environment, and a table
mapping each paper table and figure (main paper and Supporting Information)
to its archive, command, and output files. If you already have the archives,
use `--archives-dir DIR` instead of (or together with) `--revision`; missing
files are then downloaded into `DIR`. A complete run takes about 10 minutes
on a laptop. The authors' record of this procedure is in
`RELEASE_VERIFICATION.md`.

This reconstructs recorded measurements. It is not an independent replication
and does not rerun GPU training.

## Terminology: operating limit and "safe"

The operating limit is a declared convention: device capacity minus
2,048 MiB, that is, **38,912 MiB** on the 40,960 MiB A100 and 63,472 MiB on a
65,520 MiB MI250X GCD. It is not a physical threshold, a failure probability,
or a completion guarantee. A configuration is *within margin* when all
eligible repetitions complete with a sampled per-device peak at or below the
limit; *above margin* when all complete but one exceeds it; *memory failure*
when a repetition has a diagnosed device-memory failure. In field names,
`safe` is shorthand for within-margin completion only, and `unsafe` for "not
within margin" (above-margin completions and memory failures). A within-margin
configuration may still fail in a future run, and an above-margin one still
completed. See `docs/TERMINOLOGY.md` for every `safe`-family field.

## The lightweight benchmark

Predict whether a target configuration completes within margin, completes
above it, or has a diagnosed memory failure. The curated `benchmark/`
directory holds 52 tasks with 400 queries reusing 94 distinct target
configurations, not 400 independent experiments. Python 3.11+ only:

```bash
python3 benchmark.py verify
python3 benchmark.py baseline --output predictions.csv
python3 benchmark.py evaluate --predictions predictions.csv --output scores.json
python3 decision_benchmark.py replay --protocol benchmark/decision/protocol.json \
  --attempts benchmark/decision/results/attempts.json --output admission-scores.json
```

Read within-margin recall together with approval precision and the separate
approvals of memory failures and above-margin completions; pooled accuracy is
not an adequate deployment ranking. `BENCHMARK.md` gives the information
boundary and `DATA_DICTIONARY.md` the fields and units. Target outcomes are
public for inspection and are not a hidden test set.

## Repository layout

| Path | Content |
|---|---|
| `reproduce_all.sh`, `reproduce_all.py`, `reproduction/` | single entry point, expected figure/table hashes, paper map |
| `reproduce.py`, `replication_reproduction.py`, `source-provenance.json` | historical archive reconstruction (hash-bound modules) |
| `memory_tuner/`, `benchmark/`, `profiles/electronics_revision/` | analysis modules and curated tables |
| `amd_mi250x/`, `estimator_comparison/`, `third_party/` | MI250X replication, estimator comparison, upstream LLMem |
| `revision_2026_11/` | E1–E3 experiment code, curated results, CPU re-analyses |
| `docs/` | terminology, earlier release guides (`README_2026-10-23.md`, `SUBMISSION_RELEASE.md`, `REPRODUCING_ELECTRONICS_REVISION.md`), telemetry audit |
| `download.py` | optional `huggingface_hub` download helper for the five archives |
| `submission-release.json` | archive inventory of the previous (October 2026) release, kept for history |

`DATA_DICTIONARY.md`, `BENCHMARK.md` and the copied modules are hash-bound
to the historical archive in `source-provenance.json`; `reproduce.py` rejects
changed copies, so new definitions live in `docs/TERMINOLOGY.md`.

## Scope

The evidence covers the measured LoRA-GRPO stack on A100-40GB (and the
separately versioned MI250X stack), one- to 100-step runs, and the listed
models and workloads. It does not establish full-schedule convergence,
frontier-scale feasibility, multi-node scaling, or transfer to other engines.
Some historical source records identify modified working trees without
per-run patches; later panels, including E1–E3, ran from immutable source
snapshots, which the archives contain.

## License and citation

Original code, measurements, and documentation: MIT (`LICENSE`); third-party
material keeps its terms (`RIGHTS.md`). Cite the dataset with its DOI and the
version tag (`CITATION.cff`).
