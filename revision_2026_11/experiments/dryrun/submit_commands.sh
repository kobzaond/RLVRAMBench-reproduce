#!/usr/bin/env bash
# Rendered submission commands (NOT executed). Replace <...> after approval.
set -euo pipefail
# smoke1: 1 array tasks x 2 GPUs, --time=01:30:00, at most 1 concurrent
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh smoke1 /mnt/proj1/fta-26-70/rlvram/protocols/smoke1/smoke1_protocol.json <PROTOCOL_SHA256_SMOKE1> <SOURCE_COMMIT> --submit
#   which runs: sbatch --parsable --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<SOURCE_COMMIT_12>,SOURCE_COMMIT=<SOURCE_COMMIT>,PROTOCOL_PATH=...,PROTOCOL_SHA256=... experiments/ieee_resubmission_2026-11/slurm/ieee_resub_smoke1.slurm

# smoke2: 1 array tasks x 4 GPUs, --time=02:00:00, at most 1 concurrent
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh smoke2 /mnt/proj1/fta-26-70/rlvram/protocols/smoke2/smoke2_protocol.json <PROTOCOL_SHA256_SMOKE2> <SOURCE_COMMIT> --submit
#   which runs: sbatch --parsable --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<SOURCE_COMMIT_12>,SOURCE_COMMIT=<SOURCE_COMMIT>,PROTOCOL_PATH=...,PROTOCOL_SHA256=... experiments/ieee_resubmission_2026-11/slurm/ieee_resub_smoke2.slurm

# e1: 12 array tasks x 2 GPUs, --time=02:00:00, at most 3 concurrent
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh e1 /mnt/proj1/fta-26-70/rlvram/protocols/e1/e1_protocol.json <PROTOCOL_SHA256_E1> <SOURCE_COMMIT> --submit
#   which runs: sbatch --parsable --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<SOURCE_COMMIT_12>,SOURCE_COMMIT=<SOURCE_COMMIT>,PROTOCOL_PATH=...,PROTOCOL_SHA256=... experiments/ieee_resubmission_2026-11/slurm/ieee_resub_e1.slurm

# e2: 10 array tasks x 4 GPUs, --time=01:40:00, at most 3 concurrent
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh e2 /mnt/proj1/fta-26-70/rlvram/protocols/e2/e2_protocol.json <PROTOCOL_SHA256_E2> <SOURCE_COMMIT> --submit
#   which runs: sbatch --parsable --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<SOURCE_COMMIT_12>,SOURCE_COMMIT=<SOURCE_COMMIT>,PROTOCOL_PATH=...,PROTOCOL_SHA256=... experiments/ieee_resubmission_2026-11/slurm/ieee_resub_e2.slurm

# e3: 12 array tasks x 4 GPUs, --time=11:30:00, at most 3 concurrent
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh e3 /mnt/proj1/fta-26-70/rlvram/protocols/e3/e3_protocol.json <PROTOCOL_SHA256_E3> <SOURCE_COMMIT> --submit
#   which runs: sbatch --parsable --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<SOURCE_COMMIT_12>,SOURCE_COMMIT=<SOURCE_COMMIT>,PROTOCOL_PATH=...,PROTOCOL_SHA256=... experiments/ieee_resubmission_2026-11/slurm/ieee_resub_e3.slurm

# e3-amendment-01 (sha256 966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 as rendered; recompute after any edit), copied read-only to /mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-01.json
# (a) blocks never launched, array indices 5-11, under the amended snapshot
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh e3 /mnt/proj1/fta-26-70/rlvram/protocols/e3/e3_protocol.json f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de <AMENDED_SOURCE_COMMIT> --amendment /mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-01.json 966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 --indices 5-11 --submit
#   which runs: sbatch --parsable --array=5-11%3 --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<AMENDED_SOURCE_COMMIT_12>,SOURCE_COMMIT=<AMENDED_SOURCE_COMMIT>,PROTOCOL_PATH=/mnt/proj1/fta-26-70/rlvram/protocols/e3/e3_protocol.json,PROTOCOL_SHA256=f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de,AMENDMENT_PATH=/mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-01.json,AMENDMENT_SHA256=966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 <SNAPSHOT>/experiments/ieee_resubmission_2026-11/slurm/ieee_resub_e3.slurm
# (b) amendment reruns, each submitted once its original has ended unresolved:
#   rerun 0: e3-m7-code_heavy_tail-off1-r70-s631-restore (original job 5381357)
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh e3 /mnt/proj1/fta-26-70/rlvram/protocols/e3/e3_protocol.json f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de <AMENDED_SOURCE_COMMIT> --amendment /mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-01.json 966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 --rerun --indices 0 --submit
#   which runs: sbatch --parsable --array=0%3 --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<AMENDED_SOURCE_COMMIT_12>,SOURCE_COMMIT=<AMENDED_SOURCE_COMMIT>,PROTOCOL_PATH=/mnt/proj1/fta-26-70/rlvram/protocols/e3/e3_protocol.json,PROTOCOL_SHA256=f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de,AMENDMENT_PATH=/mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-01.json,AMENDMENT_SHA256=966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 <SNAPSHOT>/experiments/ieee_resubmission_2026-11/slurm/ieee_resub_e3_amendment.slurm
#   rerun 1: e3-m7-code_heavy_tail-off1-r70-s632-long (original job 5382462); after its original attempt has ended unresolved at the 21,600 s timeout (dispatched 18:33:51 UTC, so about 00:34 UTC on 11 November) and block e3-m7-code_heavy_tail-off1-r70-s632 has written summary.json
experiments/ieee_resubmission_2026-11/submit/submit_ieee_resub.sh e3 /mnt/proj1/fta-26-70/rlvram/protocols/e3/e3_protocol.json f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de <AMENDED_SOURCE_COMMIT> --amendment /mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-01.json 966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 --rerun --indices 1 --submit
#   which runs: sbatch --parsable --array=1%3 --export=ALL,IEEE_RESUB_SOURCE_ROOT=/mnt/proj1/fta-26-70/rlvram/src/ieee-resub-<AMENDED_SOURCE_COMMIT_12>,SOURCE_COMMIT=<AMENDED_SOURCE_COMMIT>,PROTOCOL_PATH=/mnt/proj1/fta-26-70/rlvram/protocols/e3/e3_protocol.json,PROTOCOL_SHA256=f6ef9c52ddae8ef97e832e85215aa79341c29f262e00d59611e088751acd53de,AMENDMENT_PATH=/mnt/proj1/fta-26-70/rlvram/protocols/e3/amendment-01.json,AMENDMENT_SHA256=966067e46423acd8dc256b313357a21a19010218243865d672127d9e88f62ee8 <SNAPSHOT>/experiments/ieee_resubmission_2026-11/slurm/ieee_resub_e3_amendment.slurm

