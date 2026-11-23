#!/usr/bin/env python3
"""Hash-chained protocol amendments (PLAN.md section 9).

An amendment is written before any rerun, chained to the frozen protocol by
its SHA-256, and copied read-only next to it as
protocols/<study>/amendment-NN.json. It records the logged non-memory defects,
the exact slots that run again, the slots that are only reassessed from raw
evidence, the blocks that start for the first time under the amended source,
and the complete source-file hashes of the amended snapshot (the frozen
protocol keeps the hashes of the original snapshot). Nothing here submits jobs.

`refresh` recomputes the source hashes of the working tree into the amendment
(run after any edit of a hashed source file, before the commit); `check`
verifies an amendment against a protocol and the current tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ieee_resub import common as C  # noqa: E402

C.bootstrap()

SCHEMA = "ieee-resub-amendment-1"
STATUS = "recorded_before_rerun"
FILE_NAME = re.compile(r"amendment-(\d{2})\.json")


def tag(amendment: dict) -> str:
    return f"amendment-{int(amendment['number']):02d}"


def changed_sources(before: dict, after: dict) -> dict:
    return {k: dict(before=before.get(k), after=after.get(k))
            for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}


def validate(amendment: dict, protocol: dict, protocol_hash: str, study: str) -> dict:
    """Structure and hash chain; independent of where the file lives."""
    C.require(amendment.get("schema") == SCHEMA and amendment.get("status") == STATUS, "Not a recorded amendment")
    C.require(amendment["study"] == study == protocol["study"] and amendment["group"] == protocol["group"],
              "Amendment belongs to another study")
    C.require(amendment["protocol_sha256"] == protocol_hash, "Amendment is not chained to this protocol")
    C.require(amendment["amendment_id"] == f"{study}-{tag(amendment)}", "Amendment identity differs")
    before, after = protocol["source_files_sha256"], amendment["source_files_sha256"]
    C.require(amendment["changed_source_files"] == changed_sources(before, after),
              "Amendment source changes are not chained to the protocol's source hashes")
    reruns = amendment["reruns"]
    C.require(len({r["experiment_id"] for r in reruns}) == len(reruns), "Duplicate rerun slot")
    for entry in reruns:
        C.require(C.SAFE_ID.fullmatch(entry["experiment_id"]) and C.SAFE_ID.fullmatch(entry["block_id"]),
                  "Unsafe rerun identity")
        C.require(entry["claim"] == f"blocks/{entry['block_id']}-{tag(amendment)}", "Rerun claim name differs")
        C.require(entry["required_original_status"] == "unresolved",
                  "Only unresolved (non-memory) slots may run again")
    launch = amendment["amended_launch_blocks"]
    C.require(len(launch["array_indices"]) == len(launch["block_ids"]) == len(set(launch["array_indices"])),
              "Amended launch list is inconsistent")
    return amendment


def load(path: Path, expected_hash: str, protocol_path: Path, protocol: dict, protocol_hash: str,
         study: str) -> dict:
    """Load and validate the hash-anchored amendment stored next to the protocol."""
    path = Path(path)
    C.require(re.fullmatch("[a-f0-9]{64}", expected_hash or "") is not None, "Invalid amendment anchor")
    C.require(FILE_NAME.fullmatch(path.name) is not None and path.parent == Path(protocol_path).parent,
              "Amendments live next to their protocol as amendment-NN.json")
    C.require(C.digest(path) == expected_hash, "Amendment changed")
    amendment = validate(json.loads(path.read_text()), protocol, protocol_hash, study)
    C.require(path.name == tag(amendment) + ".json", "Amendment file name differs from its number")
    return amendment


def matrix_row(rows: list[dict], experiment_id: str) -> dict:
    found = [r for r in rows if r["experiment_id"] == experiment_id]
    C.require(len(found) == 1, f"Amended slot {experiment_id} is not a frozen matrix row")
    return found[0]


def check_against_matrix(amendment: dict, rows: list[dict]) -> None:
    for entry in amendment["reruns"]:
        row = matrix_row(rows, entry["experiment_id"])
        C.require(row["block_id"] == entry["block_id"] and int(row["period"]) == int(entry["period"])
                  and row["role"] == entry["role"], "Rerun entry differs from the frozen matrix row")
    for entry in amendment["reassessed_slots"]:
        row = matrix_row(rows, entry["experiment_id"])
        C.require(int(row["period"]) == int(entry["period"]), "Reassessed entry differs from the matrix")
    launch = amendment["amended_launch_blocks"]
    for index, block in zip(launch["array_indices"], launch["block_ids"]):
        C.require({r["block_id"] for r in rows if int(r["block_index"]) == int(index)} == {block},
                  f"Array index {index} is not block {block}")


def refresh(path: Path, protocol: dict) -> dict:
    """Write the amended tree's source hashes into the amendment (local, before the commit)."""
    from ieee_resub.plan import source_hashes
    amendment = json.loads(Path(path).read_text())
    after = source_hashes()
    amendment["source_files_sha256"] = after
    amendment["changed_source_files"] = changed_sources(protocol["source_files_sha256"], after)
    C.write_json(Path(path), amendment)
    return amendment["changed_source_files"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("refresh", "check"):
        p = sub.add_parser(name)
        p.add_argument("--amendment", type=Path, required=True)
        p.add_argument("--protocol", type=Path, required=True, help="the frozen protocol (a read-only copy is fine)")
        p.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    C.require(C.digest(args.protocol) == args.protocol_sha256, "Protocol hash differs")
    protocol = json.loads(args.protocol.read_text())
    if args.command == "refresh":
        print(json.dumps(refresh(args.amendment, protocol), indent=2))
    amendment = validate(json.loads(args.amendment.read_text()), protocol, args.protocol_sha256, protocol["study"])
    check_against_matrix(amendment, C.read_rows(C.EXPERIMENT_DIR / "matrices" / f"{protocol['study']}_matrix.csv"))
    from ieee_resub.plan import source_hashes
    current = source_hashes() == amendment["source_files_sha256"]
    print(json.dumps(dict(amendment_id=amendment["amendment_id"], amendment_sha256=C.digest(args.amendment),
                          protocol_sha256=args.protocol_sha256, changed_source_files=sorted(
                              amendment["changed_source_files"]), tree_matches_amendment=current), indent=2))
    return 0 if current else 1


if __name__ == "__main__":
    raise SystemExit(main())
