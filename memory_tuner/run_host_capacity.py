"""Execute one frozen expanded-host-resource pair. Never submit or retry jobs."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

from memory_tuner import run_matched_gpu as runner


GROUP = "estimation_host_capacity"


def execute_pair(args):
    # Keep protocol ownership separate; --help does not need frozen inputs.
    from memory_tuner.host_capacity_protocol import (
        validate_allocation, validate_matrix, verify_protocol,
    )

    return runner.execute_pair(
        args, group=GROUP, matrix_validator=validate_matrix,
        protocol_verifier=verify_protocol, allocation_validator=validate_allocation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, _frame):
        raise InterruptedError(f"allocation signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return execute_pair(args)


if __name__ == "__main__":
    sys.exit(main())
