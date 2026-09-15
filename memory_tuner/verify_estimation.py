"""Verify archived estimator inputs, reproduce predictions, and optionally refit.

This never overwrites the pre-execution seal or reads prospective outcomes.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from memory_tuner import estimation_baselines as eb


def compare(actual, expected, path="root"):
    """Tight numerical tolerance; categorical decisions and identities are exact."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Object keys differ: {path}")
        for key in expected:
            compare(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Array shape differs: {path}")
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}[{index}]")
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (float, int))
                or not np.isfinite(actual)
                or not np.isclose(actual, expected, rtol=1e-9, atol=1e-6)):
            raise ValueError(f"Numerical value differs: {path}")
    elif actual != expected or (isinstance(expected, bool) and type(actual) is not bool):
        raise ValueError(f"Value differs: {path}")


def verify(root, refit=False):
    root = root.resolve()
    directory = root / "benchmark/estimation"
    protocol, amendments = eb.load_protocol(root, directory)
    seal = json.loads((directory / "prediction_freeze.json").read_text())
    if (seal["protocol_sha256"] != eb.sha256(directory / "protocol.json")
            or seal["amendments_sha256"] != amendments
            or seal["effective_input_sha256"] != protocol["input_sha256"]
            or set(seal["files_sha256"]) !=
                {"retrospective.json", "fitted_model.json", "predictions.json"}):
        raise ValueError("Seal does not bind the effective protocol and expected outputs")
    for name, expected in seal["files_sha256"].items():
        if eb.sha256(directory / name) != expected:
            raise ValueError(f"Sealed output differs: {name}")
    metadata = json.loads((directory / "model_metadata.json").read_text())
    source = eb.source_records(root, set(protocol["source_configuration_ids"]))
    targets = [{"settings": row} for row in eb.read_csv(directory / "targets.csv")]
    if (len(targets) != 12 or {r["settings"]["configuration_id"] for r in targets}
            != set(protocol["target_configuration_ids"])):
        raise ValueError("Target configuration coverage differs")
    fitted = json.loads((directory / "fitted_model.json").read_text())
    predictions = json.loads((directory / "predictions.json").read_text())
    retrospective = json.loads((directory / "retrospective.json").read_text())
    compare(eb.predict(targets, source, metadata, fitted), predictions, "prospective_predictions")
    compare(eb.source_cost(predictions, source), fitted["source_cost"], "prospective_source_cost")
    recomputed_scores = {}
    captured = io.StringIO()
    with redirect_stderr(captured), threadpool_limits(limits=1):
        for family, fold in retrospective["folds"].items():
            training = [r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] != family]
            testing = [r for r in source if eb.MODEL_FAMILIES[r["settings"]["model_id"]] == family]
            if len(training) != 60 or len(testing) != 30:
                raise ValueError("Held-out family split differs")
            observed = eb.predict(testing, training, metadata, fold["model"])
            compare(observed, fold["predictions"], family + ".predictions")
            scores = eb.score(observed, testing)
            compare(scores, fold["scores"], family + ".scores")
            compare(eb.source_cost(observed, training), fold["source_cost"], family + ".source_cost")
            recomputed_scores[family] = scores
            if refit:
                model = eb.fit(training, metadata)
                compare(model, fold["model"], family + ".refitted_model")
                compare(eb.predict(testing, training, metadata, model),
                        fold["predictions"], family + ".refitted_predictions")
        if refit:
            model = eb.fit(source, metadata)
            compare(model, {key: value for key, value in fitted.items()
                            if key not in ("numerical_environment", "source_cost")},
                    "prospective_refitted_model")
            compare(eb.predict(targets, source, metadata, model), predictions,
                    "prospective_refitted_predictions")
    return {
        "status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_seal_sha256": eb.sha256(directory / "prediction_freeze.json"),
        "protocol_sha256": seal["protocol_sha256"], "amendments_sha256": amendments,
        "effective_inputs_verified": len(protocol["input_sha256"]),
        "sealed_outputs_verified": len(seal["files_sha256"]),
        "retrospective_folds": 3, "retrospective_configuration_predictions_per_method": 90,
        "prospective_configuration_predictions_per_method": 12,
        "fresh_refit_compared": refit,
        "numerical_comparison": {"relative_tolerance": 1e-9, "absolute_tolerance": 1e-6,
                                 "categorical_and_identity_tolerance": 0},
        "numerical_backend_diagnostics": captured.getvalue(),
        "thread_limit_interpretation": (
            "One thread was requested through threadpoolctl. A backend discovery warning "
            "means that not every loaded library's thread count was established. "
            "This verification tests numerical and categorical agreement, not CPU performance."),
        "prospective_outcomes_read": False,
        "retrospective_scores": recomputed_scores,
        "source_cost": fitted["source_cost"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    result = verify(args.root, args.refit)
    eb.write_json(args.output, result)
    print(json.dumps({key: result[key] for key in
                      ("status", "fresh_refit_compared", "sealed_outputs_verified",
                       "prospective_outcomes_read")}, indent=2))


if __name__ == "__main__":
    main()
