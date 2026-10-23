# Electronics submission release — 23 October 2026

The paper, dataset, and reproduction code use the version tag
`electronics-submission-2026-10-23-r2`. The submission dataset is published on
the Hugging Face `main` branch and pinned with the same tag:
<https://huggingface.co/datasets/kobzaond/RLVRAMBench>.
The dataset DOI is `10.57967/hf/10780`; use the version tag to identify
this particular submission. Earlier release tags and archives are retained.

`submission-release.json` inventories the evidence archives. The root
`release.json` remains the historical A100 reconstruction contract used by
`reproduce.py`; it is not the version selector for the whole submission.
No model weights or optimizer checkpoints are distributed.

## Download the submission data

The direct Hub-loading interface is tested with Python 3.11,
`datasets==5.1.0` and `huggingface_hub==2.2.0`. Install these from
the dataset's `dataset-client-requirements.txt` in a separate reader environment. Older
clients such as `datasets==2.12.0` ignore the card's configuration mapping
and may try to combine unrelated audit JSON files. The frozen raw-analysis
environment is documented separately in the reproduction guide.

```sh
python -m pip install -r dataset-client-requirements.txt
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "kobzaond/RLVRAMBench",
    repo_type="dataset",
    revision="electronics-submission-2026-10-23-r2",
    local_dir="RLVRAMBench-data",
)
```

The dataset card exposes separate CSV views for historical measurements,
independent controls, A100 follow-ups, MI250X trials and configurations,
attempt accounting, and estimator comparisons. Cohorts must not be pooled
as independent experiments: the 400 benchmark queries reuse 94 targets;
repeated seeds, paired runs, and multiple estimator methods also reuse
experimental units.

## Historical A100 evidence

Use the archive named by the root `release.json`:

```sh
python reproduce.py \
  --archive RLVRAMBench-data/releases/RLVRAMBench-review-artifact-submission-20260917-b0018d0.tar.zst \
  --work-dir historical-reconstruction --isolate-analysis
```

This reconstructs the historical measurements, admission and estimation
panels, and the completion and instrumentation replications. Follow the
root README for the pinned CPU environment and isolation options.

## Independent offload controls

The archive includes the 50,058 hash-verified raw evidence files, the
frozen execution source, and terminal allocation accounting.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir offload-raw
tar --use-compress-program=unzstd \
  -xf RLVRAMBench-data/releases/RLVRAMBench-offload-raw-20261022.tar.zst \
  -C offload-raw
(cd offload-raw && sha256sum -c --quiet MANIFEST.sha256)
python -m memory_tuner.collect_offload_factorial \
  --root offload-raw --source offload-raw/source \
  --freeze-sha256 5df43f35e0b02c23e4816b7e860c5a613176da41f5adb27e57f79ac2becf1fb9 \
  --source-commit 3d6ecac1852e16a0422c9c72a230187be663f8b5 \
  --accounting offload-raw/allocation_accounting.json \
  --output offload-reconstructed
```

The five CSV outputs match the released tables byte for byte. The summary
comparison permits only the documented `raw_root` and `source_snapshot`
directory changes.

## A100 Electronics follow-ups and native LLMem

Extract `releases/electronics-full-20261021-a53af63.tar.zst` and follow
`docs/REPRODUCING_ELECTRONICS_REVISION.md`. The archive carries its own
verification script:

```sh
mkdir a100-followups
tar --use-compress-program=unzstd \
  -xf RLVRAMBench-data/releases/electronics-full-20261021-a53af63.tar.zst \
  -C a100-followups
python a100-followups/paper/verify_electronics_reconstruction.py \
  a100-followups --report a100-followups-verification.json
```

The evidence manifest has 112,689 files. The verifier reconstructs 46
outputs, preserving all four ineligible formal RL outcomes and the
separate development records. It does not rerun training.

## MI250X replication

```sh
mkdir mi250x-raw
tar --use-compress-program=unzstd \
  -xf RLVRAMBench-data/releases/RLVRAMBench-mi250x-raw-20261022.tar.zst \
  -C mi250x-raw
(cd mi250x-raw && sha256sum -c --quiet release/MANIFEST.sha256)
python amd_mi250x/collect.py --root mi250x-raw --output mi250x-reconstructed
python amd_mi250x/analyze.py --results mi250x-reconstructed
python amd_mi250x/make_supplement.py \
  --results mi250x-reconstructed --output supplement_mi250x.tex
```

The 462 planned slots yield 363 completions, 98 diagnosed memory failures,
and one ineligible non-memory outcome. All 461 eligible runs have matching
memory/phase traces, and all completions record every requested step and
exit successfully. The 90-configuration A100 replay has 79 agreements among
89 resolved labels, plus one unresolved configuration. Its two observed
OOMs remain adverse evidence. The attempt ledger distinguishes 510 started
jobs from 511 executions because one job restarted after node failure.
The four amendments and original attempts remain in the release.

## Estimators and paper

```sh
python estimator_comparison/run.py --output estimator-reconstructed
python estimator_comparison/make_supplement.py \
  --results estimator-reconstructed --output supplement_deepspeed.tex
python -m pytest -q tests
```

Replaying recorded DeepSpeed calls requires no GPU or live DeepSpeed
installation. The native LLMem validation belongs to the separate A100
archive. Neither comparison establishes an end-to-end estimator for every
colocated RL stack.

The 23 October release changes presentation only. Figure-summary CSV files
and all main-paper table rows match the preceding release. The updated
renderers use readable labels within the manuscript text column; the wide
MI250X inventory uses landscape pages with repeated headers.

To render the revised historical boundary and logging plots from the
curated files downloaded above:

```sh
mkdir -p submission-figures
python -m memory_tuner.electronics_historical_figures \
  --root . --data RLVRAMBench-data/tables \
  --review-results RLVRAMBench-data/tables/review_revision --output submission-figures
python -m memory_tuner.electronics_followup_figures \
  --root . --figures submission-figures
python -m memory_tuner.electronics_revision_tables \
  --root . --output submission-tables
```

The frozen historical reconstruction command still uses its original,
hash-verified plotting module. `verification/layout-audit.json` records
checks of the submission PDFs, and `verification/layout-data-preservation.json`
records the numerical comparisons for the revised figures and tables.

The dataset's `paper/` directory contains the main manuscript, supplement,
reviewer response, and editable TeX ZIP. `verification/` records the release
checks. The ZIP README gives the standalone build commands.

These are author-performed reproduction checks. The evidence does not
establish full-schedule convergence, frontier-model feasibility, multi-node
scaling, or transfer to hardware and engines outside the measured stacks.
