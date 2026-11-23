# Retrospective CPU re-analyses (November 2026)

CPU-only re-analyses of already released measurements, written after the
outcomes were known. They are diagnostics, not frozen predictions; nothing
here runs on a GPU or reruns training.

| Directory | Analyses | Paper (SI numbering of the CCPE version) |
|---|---|---|
| `c1_realized_work/` | C1: prompt/response lengths, cap reaching and token volume behind the micro-batch contrast | Section S27, Table S54; Table 10 cross-check |
| `c2_predictors/` | C2 failure-aware hurdle predictors, C3 coverage refit, C8 margin sensitivity of the baselines | Section S30, Tables S65–S68; Table 15 |
| `c4_sensitivity/` | C4 eligibility-rule sensitivity, C5 case-level uncertainty, C6 GPU time in budgeted admission | Sections S28–S29, Tables S55–S64 |

Each directory holds the scripts and `reference/`, the outputs reported in the
paper. `../../reproduce_all.sh` reruns all of them and compares every output
with `reference/` (only run-specific file locations in C1's `REPORT.md` and
`key_results.json` may differ).

## Inputs and environment

The scripts read a *tables root* with the layout of the research repository:
`benchmark/`, `benchmark.py`, `memory_tuner/` (the code files hashed by the
frozen protocols), `profiles/electronics_revision/`, `profiles/standard_grpo/`
and `profiles/review_revision/`. `reproduce_all.py` assembles it from this
repository and from the historical archive (`reference-profiles/` written by
`reproduce.py`), and records in `cpu_reanalyses/inputs.csv` which input
files were regenerated from raw evidence in the same run. C1 also reads the
extracted historical archive.

Environment variables (set by `reproduce_all.py`):

- `RLVRAMBENCH_TABLES_ROOT`: the tables root;
- `RLVRAMBENCH_ANALYSIS_OUTPUT`: a new output directory for one group;
- `RLVRAMBENCH_CODE_ROOT` (optional): code checkout providing `memory_tuner/`.

The reported values were produced with CPython 3.12 and the versions pinned
in `requirements.txt` (numpy 2.1.2, scipy 1.17.1, scikit-learn 1.9.0).
Under the historical publication environment (CPython 3.11,
`publication-requirements.txt`) most outputs are identical, but floating-point
summation and the logistic-regression solver differ slightly; `reproduce_all.py`
lists those files in an informational step. Under that older stack, for
example, the leave-one-out sensitivity of the refitted classifier (Section
S28) changes for a few deletions.

Manual run of one group:

```sh
export RLVRAMBENCH_TABLES_ROOT=/path/to/tables-root RLVRAMBENCH_ANALYSIS_OUTPUT=/path/to/out/c4
python3.12 -I c4_sensitivity/c4_eligibility.py   # then c4_predictors, c5_uncertainty, c6_gpu_time
```
