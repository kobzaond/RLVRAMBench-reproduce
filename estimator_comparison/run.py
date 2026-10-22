#!/usr/bin/env python3
"""Score the analytic and DeepSpeed estimators on all RLVRAMBench queries.

Usage (from the repository root):

    .venv-reproduce/bin/python estimator_comparison/run.py \
        --data benchmark --output estimator_comparison/results

Writes predictions, ``benchmark.py evaluate`` scores, peak-error tables and a
figure. CPU only and deterministic. DeepSpeed itself is not imported here; its
recorded outputs are read from ``estimator_comparison/deepspeed_estimates.json``
(produced by ``deepspeed_collect.py`` in a separate environment).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from estimator_comparison import configurations, deepspeed_baseline, estimator_eval  # noqa: E402
from estimator_comparison.common import Benchmark, write_csv, write_json  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "benchmark")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = args.data if args.data.is_absolute() else Path.cwd() / args.data
    out = args.output if args.output.is_absolute() else Path.cwd() / args.output
    out.mkdir(parents=True, exist_ok=True)
    bench = Benchmark(data)
    repo_root = data.resolve().parent

    configs = configurations.collect(bench)
    write_csv(out / "completed_and_failing_configurations.csv",
              [{k: v for k, v in c.items() if k != "settings"} | c["settings"] for c in configs])
    summary = {
        "analytic": estimator_eval.run(bench, out, out / "figures", repo_root, configs),
        "deepspeed": deepspeed_baseline.run(bench, out, repo_root, configs),
    }
    write_json(out / "summary.json", summary)
    print(f"Wrote estimator comparison to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
