# C1: realized work behind the micro-batch contrast

`c1_realized_work.py` tabulates prompt and response lengths, cap reaching,
and token volume for every process of the same-device replication (144
invocations) and the original factorial experiment (144 processes), sets the
within-block token difference between micro-batches 16 and 8 next to the
actor-stage contrast, and fits a descriptive within-case adjustment. It first
reproduces the published equal-case means (3,905 / 7 MiB and 3,980 / −31 MiB)
and stops if they differ.

```sh
python3.12 -I c1_realized_work.py --repo TABLES_ROOT --extract HISTORICAL_ARTIFACT \
  --archive RLVRAMBench-review-artifact-submission-20260917-b0018d0.tar.zst --out OUT
```

`HISTORICAL_ARTIFACT` is the extracted historical archive (for example
`historical/artifact/` of a `reproduce.py` work directory); only the 144
`trainer-metrics/step-1.json` records and 144 factorial training logs are
read. Outputs: `rc_*`/`of_*` CSV tables, `reproduction_check.csv`,
`key_results.json`, the two SI table fragments, and `REPORT.md`.
