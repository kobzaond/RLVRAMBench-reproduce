# Revision experiments E1–E3 (November 2026)

Three frozen A100 experiments run on Karolina (A100-SXM4-40GB) in
November 2026, after the earlier releases:

| Study | What it tests | Planned slots | Outcome |
|---|---|---:|---|
| E1 controlled-length experiment | Actor micro-batch 8 vs 16 with fixed response length and canonical batch order, under external-only and full instrumentation | 48 | 48 within margin |
| E2 near-boundary repetitions | Ten fresh one-step runs of four configurations near the operating limit | 40 | 30 within margin, 10 above margin |
| E3 7B duration and restore test | One-step screen, 100-step run with validation and checkpoints, and restore from step 50 | 28 (30 attempts) | 28 within margin |

Two development smoke gates (`smoke1`, `smoke2`) preceded the formal runs and
are excluded from every estimate.

## Contents

- `experiments/`: a byte-identical copy of `experiments/ieee_resubmission_2026-11/`
  of the research repository at commit `cc685c09a446c62f4507432e1962c2f49b277b1d`
  (frozen protocol builder, launcher, runner, collector, analysis, Slurm scripts,
  VERL patch, instrumented packages, matrices, dry run, tests, `PLAN.md`,
  `AMENDMENTS.md`). The directory name is part of the frozen source hashes, so
  it is kept unchanged inside the snapshots. `experiments/results/e1/analysis`
  is the pre-amendment E1 analysis (see below); the reported results are in
  `results/`.
- `results/<study>/`: collector outputs (`summary.json`, `outcomes.json`,
  `outcomes.csv`, and for E3 the attempt ledger `attempts.json`/`.csv`) as
  written on the cluster, and `analysis.json` from `ieee_resub/analysis.py`
  of commit `cc685c0`. `raw_path` values keep their cluster location
  (`/mnt/proj1/fta-26-70/rlvram/output/...`).
- `cpu_reanalyses/`: the retrospective CPU re-analyses C1, C2/C3/C8 and
  C4–C6, with their reference outputs.

The raw evidence is the dataset archive
`releases/RLVRAMBench-revision-raw-20261123.tar.zst` (see `../release.json`).
It contains every invocation directory, block claim, terminal receipt,
frozen protocol, submission receipt, amendment A7, collector output, and
setup/job log, plus `source/<commit>/`: the hash-bound executing files of
the two immutable snapshots that ran the studies (`9f6faf9` for E1, E2, the
smoke gates and E3 blocks 0–4; `cc685c0` for E3 blocks 5–11 and the two
amendment reruns), together with the import closure of the collector and
analysis. Checkpoints, the container image, model caches and the prepared
training data are not included; their hashes are recorded in the protocols.

## Reconstruction

`../reproduce_all.sh` runs these steps; by hand, after extracting the archive
into `RAW` and checking `sha256sum -c MANIFEST.sha256` from inside `RAW`:

```sh
S9=RAW/source/9f6faf94b472f62f910858be838710aedb3cbc55/experiments/ieee_resubmission_2026-11
SC=RAW/source/cc685c09a446c62f4507432e1962c2f49b277b1d/experiments/ieee_resubmission_2026-11
python3.11 -I $S9/ieee_resub/collect.py --study e1 \
  --protocol RAW/protocols/e1/e1_protocol.json \
  --protocol-sha256 339c40d662d9948853d354e1ab133ceb49aae93a8c5989920e52f9d0dd514985 \
  --source-commit 9f6faf94b472f62f910858be838710aedb3cbc55 \
  --output-root RAW/output --results out/e1
python3.11 -I $SC/ieee_resub/collect.py --study e3 \
  --protocol RAW/protocols/e3/e3_protocol.json \
  --protocol-sha256 f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de \
  --source-commit 9f6faf94b472f62f910858be838710aedb3cbc55 \
  --amendment RAW/protocols/e3/amendment-01.json \
  --amendment-sha256 966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 \
  --amended-source-commit cc685c09a446c62f4507432e1962c2f49b277b1d \
  --output-root RAW/output --results out/e3
python3.11 -I $SC/ieee_resub/analysis.py e1 --outcomes out/e1/outcomes.json --out out/e1/analysis.json
```

E2, `smoke1` and `smoke2` are collected like E1 (protocol hashes in
`../release.json`), and E2/E3 are analyzed like E1. Each collector refuses to
run unless the executing source equals the hashes frozen in the protocol (or,
for E3, in the hash-chained amendment), so E1, E2 and the smoke gates are
collected from the `9f6faf9` snapshot and E3 from `cc685c0`. The collector
reassesses every planned slot from raw evidence and must agree with the
terminal receipts; nothing is replaced. Outputs equal `results/` after
relocating `raw_path`.

The E1 analysis written on the cluster (`results/e1/analysis` inside the
archive) predates amendment A7 defect 3, which maps the instrumentation level
`external` to the estimand level `ext`; it therefore reports no external-only
contrasts. The reported E1 analysis is the `cc685c0` re-run
(`results/e1/analysis.json`): equal-case actor-stage batch contrasts of
6,369 MiB (external-only) and 6,358 MiB (full), interaction −11 MiB.
`analysis.py check-existing` additionally reproduces earlier reported values
(3,905 / 7 / 4,060 MiB, logging-control contrasts, paired-duration growth,
release-anchor bounds) from the tables regenerated by the other archives.

GPU reruns need a new allocation, new identifiers, and a newly frozen
protocol; the block claims are permanent by design.
