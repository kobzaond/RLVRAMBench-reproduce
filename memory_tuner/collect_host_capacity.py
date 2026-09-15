"""Independently collect the separate expanded-host panel; never fit predictors.

The CLI matches collect_estimation_study. --protocol-sha256 anchors the new
host-capacity protocol; --prediction-seal-sha256 still anchors the ORIGINAL
prediction seal. Both original sealed inputs and the new study freeze are
validated before any follow-up raw outcomes are read. Outputs are the same
13-file schema, with only the follow-up slots and costs.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from memory_tuner import collect_estimation_study as core
from memory_tuner import host_capacity_protocol as design
from memory_tuner.device_contract import digest


def load_frozen(evidence, protocol_sha256=None, prediction_seal_sha256=None,
                protocol_evidence=None):
    # The original loader validates the amendment, all effective input hashes,
    # predictor implementation, original seal, and its three sealed outputs.
    original_evidence = core.Evidence(core.inside(evidence.root, "benchmark/estimation"))
    original = core.load_frozen(
        evidence, design.ORIGINAL_PROTOCOL_SHA256,
        prediction_seal_sha256 or design.ORIGINAL_SEAL_SHA256,
        protocol_evidence=original_evidence)
    original_evidence.verify_unchanged()
    for name, record in original_evidence.hashes.items():
        path = evidence.touch(original_evidence.root / name)
        core.require(evidence.hashes[path.relative_to(evidence.root).as_posix()] == record,
                     "Original protocol evidence changed during handoff")
    directory_evidence = protocol_evidence or core.Evidence(
        core.inside(evidence.root, design.DIRECTORY))
    directory = directory_evidence.root
    protocol = directory_evidence.json(directory / "protocol.json")
    for name in ("study_freeze.json", "matrix.csv", "targets.csv", "predictions.json"):
        directory_evidence.touch(directory / name)
    core.require(set(protocol["implementation_sha256"]) == set(design.IMPLEMENTATIONS),
                 "Incomplete follow-up implementation closure")
    for name in protocol["implementation_sha256"]:
        evidence.touch(core.inside(evidence.root, name))
    core.require(digest(Path(design.__file__)) ==
                 protocol["implementation_sha256"]["memory_tuner/host_capacity_protocol.py"],
                 "Executed follow-up design validator differs from frozen source")
    core.require(digest(Path(core.runner.__file__)) ==
                 protocol["implementation_sha256"]["memory_tuner/run_matched_gpu.py"],
                 "Executed period validator differs from frozen follow-up source")
    protocol, freeze, rows, targets, predictions = design.load_design(
        evidence.root, directory, protocol_sha256)
    core.require(core.epoch(freeze["frozen_at_utc"]) >=
                 core.epoch(original["seal"]["frozen_at_utc"]),
                 "Follow-up freeze predates original prediction seal")
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    # Metadata redirects only collection and validation. It cannot change any
    # inherited prediction or mix the original slots into the new denominator.
    result = dict(original, protocol=protocol, protocol_sha256=digest(directory / "protocol.json"),
                  amendments_sha256={}, rows=rows, pairs=dict(pairs), targets=targets,
                  predictions=predictions, matrix_sha256=digest(directory / "matrix.csv"),
                  group=protocol["group"], protocol_relative_dir=design.DIRECTORY,
                  evaluation_seeds=protocol["evaluation_seeds"], study_freeze=freeze,
                  allocation_validator=design.validate_allocation,
                  study_provenance=dict(
                      study_group=protocol["group"], protocol_relative_dir=design.DIRECTORY,
                      evaluation_seeds=protocol["evaluation_seeds"],
                      study_freeze_sha256=digest(directory / "study_freeze.json"),
                      study_frozen_at_utc=freeze["frozen_at_utc"],
                      original_protocol_sha256=original["protocol_sha256"],
                      original_amendments_sha256=original["amendments_sha256"],
                      original_prediction_seal_sha256=original["prediction_seal_sha256"],
                      followup_execution_implementation_sha256=protocol["implementation_sha256"],
                      study_loader_implementation_sha256=digest(Path(__file__)),
                      study_design_validation_implementation_sha256=digest(Path(design.__file__)),
                      prediction_inheritance="Exact original sealed predictions with new configuration IDs; "
                                             "no fitting or outcome-dependent update.",
                      panel_scope="Separate host-resource follow-up; no pooling with original slots."))
    # Needed when this loader is used independently rather than via core.collect.
    evidence.verify_unchanged()
    directory_evidence.verify_unchanged()
    return result


def collect(root, *, source_root=None, protocol_dir=None, **kwargs):
    source_root = Path(source_root or root).resolve()
    return core.collect(root, source_root=source_root,
                        protocol_dir=protocol_dir or core.inside(source_root, design.DIRECTORY),
                        study_frozen_loader=load_frozen, **kwargs)


def main():
    core.main(study_frozen_loader=load_frozen, protocol_relative_dir=design.DIRECTORY)


if __name__ == "__main__":
    main()
