#!/usr/bin/env python3
"""Audit the standard-GRPO RLVRAMBench manuscript and submission package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from memory_tuner.build_submission_manuscript import (
    ABSTRACT_WORD_LIMIT,
    abstract_word_count,
    validate_figure_resources,
    validate_highlights,
    validate_manuscript,
)
from memory_tuner.check_paper_citations import audit as citation_audit


TITLE = (
    "RLVRAMBench: A Failure-Inclusive Benchmark for GPU-Memory "
    "Feasibility in Colocated LoRA GRPO"
)
BANNED_SCOPE_PATTERNS = {
    "nonstandard method name": r"oracle[-_ ]?sppo|\bsppo\b",
    "superseded method framing": r"(?-i:\bPhaseGuard\b)",
    "superseded serving-transfer framing": (
        r"serving-to-training|serving-safe|standalone-serving transfer"
    ),
}
REQUIRED_SUMMARY_FACTS = {
    ("scope", "algorithm"): "grpo",
    ("scope", "principal_fresh_processes"): 588,
    ("scope", "historical_fresh_processes"): 438,
    ("scope", "supporting_historical_source_processes"): 16,
    ("scope", "supporting_historical_source_configurations"): 6,
    ("scope", "evidence_status"): "complete_frozen_revision",
    ("scope", "model_families"): 3,
    ("scope", "workloads"): 4,
    ("boundary", "processes"): 216,
    ("boundary", "configurations"): 72,
    ("same_node", "pairs"): 18,
    ("factorial", "processes"): 144,
    ("temporal_100", "runs"): 12,
    ("telemetry", "downsampling_checks"): 222,
    ("revision", "processes"): 150,
    ("revision", "allocation_manifests"): 102,
    ("revision", "all_frozen_rows_and_provenance_validated"): True,
}
REQUIRED_FIGURES = (
    "standard_grpo_boundary",
    "standard_grpo_mechanisms",
    "standard_grpo_temporal",
    "standard_grpo_phase_effects",
)
AUTHOR_BLOCKERS = (
    "Final author order, affiliations, postal addresses, and corresponding "
    "author details",
    "CRediT roles, funding, competing interests, and author approval",
    "Review artifact URL, permanent DOI, and repository-level license",
    "Final referee suggestions and submission-system data statement",
)


def nested_value(data: dict, path: tuple[str, str]):
    return data.get(path[0], {}).get(path[1])


def audit_package(root: Path) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    notes: list[str] = []
    manuscript = root / "paper/phaseguard_paper_draft.md"
    bibliography = root / "paper/references.bib"
    highlights = root / "paper/submission/highlights.txt"
    summary_path = root / "profiles/standard_grpo/summary.json"

    try:
        validate_manuscript(manuscript)
        validate_highlights(highlights)
        figure_count = validate_figure_resources(manuscript)
    except ValueError as error:
        errors.append(str(error))
        figure_count = 0

    manuscript_text = manuscript.read_text(encoding="utf-8")
    for label, pattern in BANNED_SCOPE_PATTERNS.items():
        if re.search(pattern, manuscript_text, flags=re.IGNORECASE):
            errors.append(f"{label} remains in the manuscript")

    for stem in REQUIRED_FIGURES:
        for suffix in (".png", ".pdf"):
            path = root / "paper/figures" / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty figure: {path.relative_to(root)}")

    citations = citation_audit(
        manuscript_text,
        bibliography.read_text(encoding="utf-8"),
    )
    if citations["missing"]:
        errors.append(f"missing citation entries: {sorted(citations['missing'])}")
    if citations["unused"]:
        errors.append(f"unused bibliography entries: {sorted(citations['unused'])}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for path, expected in REQUIRED_SUMMARY_FACTS.items():
        observed = nested_value(summary, path)
        if observed != expected:
            errors.append(
                f"summary fact {'.'.join(path)} is {observed!r}, "
                f"expected {expected!r}"
            )

    required_docs = (
        "paper/submission/JSS_cover_letter_draft.md",
        "paper/submission/JSS_submission_checklist.md",
        "paper/submission/title_page_draft.md",
    )
    for relative in required_docs:
        text = (root / relative).read_text(encoding="utf-8")
        if re.sub(r"\s+", " ", TITLE) not in re.sub(r"\s+", " ", text):
            errors.append(f"{relative} does not contain the canonical title")
        for label, pattern in BANNED_SCOPE_PATTERNS.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                errors.append(f"{relative}: {label}")

    claim_ledger = (root / "paper/claim_ledger.md").read_text(encoding="utf-8")
    if "C10" not in claim_ledger or "Not evaluated by design" not in claim_ledger:
        errors.append("claim ledger must retain the learning-quality non-claim")
    if "Pending completion" in claim_ledger or "**Failed" in claim_ledger:
        errors.append("claim ledger contains a pending or failed scientific claim")

    notes.extend(AUTHOR_BLOCKERS)
    metrics = {
        "abstract_words": abstract_word_count(manuscript_text),
        "abstract_limit": ABSTRACT_WORD_LIMIT,
        "cited_references": len(citations["cited"]),
        "bibliography_entries": len(citations["available"]),
        "manuscript_figures": figure_count,
        "scientific_errors": len(errors),
        "author_blockers": len(notes),
    }
    return errors, notes, metrics


def render(errors: list[str], notes: list[str], metrics: dict[str, int]) -> str:
    status = "PASS" if not errors else "FAIL"
    lines = [
        "# Standard-GRPO submission-readiness audit",
        "",
        f"**Automated package-consistency status: {status}**",
        "",
        "This audit distinguishes scientific/package errors from information "
        "that only the authors can supply.",
        "",
        "## Automated checks",
        "",
        f"- Abstract: {metrics['abstract_words']}/{metrics['abstract_limit']} words",
        f"- Citation set: {metrics['cited_references']} cited / "
        f"{metrics['bibliography_entries']} bibliography entries",
        f"- Manuscript figures: {metrics['manuscript_figures']}",
        f"- Scientific/package errors: {metrics['scientific_errors']}",
        "",
        "## Scientific or package errors",
        "",
    ]
    lines.extend([f"- {error}" for error in errors] or ["- None"])
    lines.extend(
        [
            "",
            "## Author-supplied upload blockers",
            "",
            *[f"- {note}" for note in notes],
            "",
            "A PASS checks internal consistency and completeness, not scientific "
            "soundness, equivalence, generalization, or journal acceptance. "
            "It does not certify authorship, conflicts, licensing, archival "
            "deposition, or journal-system metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/submission_readiness.md"),
    )
    args = parser.parse_args()
    errors, notes, metrics = audit_package(args.root)
    report = render(errors, notes, metrics)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
