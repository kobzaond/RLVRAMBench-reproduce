# Verified release `rlvrambench-2026-11`

Author-performed verification on 23 November 2026. This is a reconstruction
of recorded measurements from the released evidence. It is not an external
replication, and it does not rerun GPU training. Earlier verification records
are in the git history and in the dataset's `verification/` directory.

| Item | Identifier |
|---|---|
| Version tag (dataset and code) | `rlvrambench-2026-11` |
| Dataset | `kobzaond/RLVRAMBench`, commit `DATASET-COMMIT-TO-BE-SET` |
| Dataset DOI | `10.57967/hf/TO-BE-MINTED` |
| Code verified | commit `3e45491b0502708516c041a9c4db43e8006416af` (branch `release-2026-11`) |
| Later code commits | add this file and fill the dataset commit, DOI, and date into `release.json`, `CITATION.cff`, `README.md`, and this file; no code changes |

## Procedure and environment

The test started from a fresh `git clone` of commit `3e45491`. The command ran
under `env -i` with `PATH=/usr/bin:/bin`. The two virtual environments were
new, and pip ran with its cache disabled:

```sh
bash reproduce_all.sh --revision 00e61d8aa329cee64c1cf899f11678971b5115e0 \
  --archives-dir archives --work-dir reproduce-all-output
```

The four earlier archives were downloaded anonymously from the dataset at
commit `00e61d8…`, which is the parent of this release. The revision archive
was not yet uploaded, so it was supplied locally. All five matched the sizes and
SHA-256 values in `release.json`:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `RLVRAMBench-review-artifact-submission-20260917-b0018d0.tar.zst` | 210,126,067 | `3dd90eaca2f2d9b5eceb955fe67b6a3528210a6bb9bd7684ac3eb3c36d84a5ab` |
| `RLVRAMBench-offload-raw-20261022.tar.zst` | 33,356,651 | `ac2c69bae8d7e0af07e27de9c182e095fa77a28f5b0d0fc906fbfff302040041` |
| `electronics-full-20261021-a53af63.tar.zst` | 74,609,840 | `f70f639fc0c991e79b9e140d4c26482a9c3490000abe3c5acf6bc62fb7a599bf` |
| `RLVRAMBench-mi250x-raw-20261022.tar.zst` | 58,527,374 | `400f75b1c3015d9903b3e898b0ed2dc3771b07363bff7f8761aec0c798ed67ae` |
| `RLVRAMBench-revision-raw-20261123.tar.zst` | 49,842,735 | `87dada01f6f7d25e01210bbfc24b7faf35041e96e3731e8bce9e24766106fe5d` |

Environment:

- Linux 6.17 x86_64, glibc 2.39, 24 CPUs;
- GNU tar 1.35, zstd 1.5.5, bubblewrap 0.9.0;
- CPython 3.11.18 with `publication-requirements.txt`;
- CPython 3.12.16 with `revision_2026_11/cpu_reanalyses/requirements.txt`
  (numpy 2.1.2, scipy 1.17.1, scikit-learn 1.9.0).

Thread counts were fixed to 1. Total wall time was 4 min 29 s, including
environment creation; the reproduction steps took 247 s. The work directory
needs 5.3 GB. The checkout stayed clean.

## Results

| Step | Status | Seconds | Checks |
|---|---|---:|---:|
| archives: obtain and verify the five archives | passed | 6.6 | 5/5 |
| historical: `reproduce.py --isolate-analysis` | passed | 110.1 | 10/10 |
| offload: `collect_offload_factorial` | passed | 26.0 | 7/7 |
| a100: archive verifier, reconstruction, renderers | passed | 56.7 | 7/7 |
| mi250x: collect, analyze, supplement | passed | 6.4 | 15/15 |
| estimator: DeepSpeed replay, analytic budget | passed | 3.2 | 3/3 |
| revision: E1–E3 and smoke-gate collectors, analyses, earlier-value checks | passed | 21.4 | 48/48 |
| cpu_reanalyses: C1, C2/C3/C8, C4–C6 | passed | 7.0 | 1/1 (72 files) |
| cpu_reanalyses_publication_env (informational) | 14 of 72 files differ | 4.5 | – |
| figures: ten paper figures, byte for byte | passed | 1.3 | 10/10 |
| tests: `pytest -q tests` | passed (196 tests, 1 skipped) | 3.0 | 1/1 |

Highlights:

- **historical:** 107,897 evidence files verified; 40 original and 27 audit/control
  files, 14 curated benchmark files, and 11 admission, 13 estimation, and
  13 expanded-host outputs matched; four figures regenerated, with references
  hidden in bubblewrap namespaces.
- **offload:** five CSVs byte-identical; the summary differs only in `raw_root` and
  `source_snapshot`.
- **a100:** 112,689 manifest files verified and 46 outputs matched by the
  archive's own verifier. The persistent reconstruction matched 42 outputs
  with the saved results unreadable, and the four SI tables equal the
  released generator output.
- **mi250x:** trials, configurations, summary, and nine analysis tables
  byte-identical; the SI section equals the released generator output.
- **revision:** collector outputs equal the archived and curated outputs after
  relocating `raw_path`. The E2/E3 analyses equal the archived ones. The E1
  analysis (re-run with the `cc685c0` analysis code, amendment A7 defect 3)
  equals `revision_2026_11/results/e1/analysis.json`. `check-existing`
  reproduced every earlier value it checks.
- **cpu_reanalyses:** all 72 outputs equal the references. Of 56 input files
  read, 34 were regenerated in the same run.
- **figures:** the ten manuscript figure PDFs (Figures 2–8 and S1–S3) are
  reproduced byte for byte.

`reproduce-all-output/REPORT.md` maps every table and figure of the paper
and its Supporting Information to the archive, command, and output files.
The report of this run is in the dataset under `verification/rlvrambench-2026-11/`.

The informational rerun of the re-analyses under the older publication stack
changes floating-point details, mainly in logistic-regression refits. In one
case the change reaches a reported count: the leave-one-out sensitivity in
the eligibility analysis. The reported values come from the pinned 3.12
environment.

After publication, the author repeats the procedure from the tagged commit
without `--revision`, which downloads all five archives at the recorded
dataset commit.
