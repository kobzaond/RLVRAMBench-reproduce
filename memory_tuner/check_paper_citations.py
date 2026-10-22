#!/usr/bin/env python3
"""Check Pandoc-style citation-key consistency against a BibTeX database.

This deliberately does not claim to validate bibliographic metadata against
Crossref, arXiv, publishers, or other primary records.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CITATION_BLOCK_RE = re.compile(r"\[[^\]]*@[^\]]+\]")
CITATION_KEY_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9:_-]*)")
BIB_KEY_RE = re.compile(
    r"@\w+\s*\{\s*([^,\s]+)\s*,",
    flags=re.IGNORECASE,
)


def citation_keys(text: str) -> set[str]:
    """Return all citation keys from bracketed Pandoc citations."""
    keys: set[str] = set()
    for block in CITATION_BLOCK_RE.findall(text):
        keys.update(CITATION_KEY_RE.findall(block))
    return keys


def bibliography_keys(text: str) -> set[str]:
    """Return entry keys from a BibTeX database."""
    return set(BIB_KEY_RE.findall(text))


def audit(manuscript_text: str, bibliography_text: str) -> dict[str, set[str]]:
    cited = citation_keys(manuscript_text)
    available = bibliography_keys(bibliography_text)
    return {
        "cited": cited,
        "available": available,
        "missing": cited - available,
        "unused": available - cited,
    }


def render(result: dict[str, set[str]], manuscript: Path, bibliography: Path) -> str:
    lines = [
        "# Citation-key consistency audit",
        "",
        f"- Manuscript: `{manuscript}`",
        f"- Bibliography: `{bibliography}`",
        f"- Cited keys: {len(result['cited'])}",
        f"- Bibliography entries: {len(result['available'])}",
        f"- Missing entries: {len(result['missing'])}",
        f"- Unused entries: {len(result['unused'])}",
        "- Scope: key resolution only; titles, authors, venues, dates, and "
        "identifiers require a separate primary-source metadata audit.",
        "",
        "## Missing entries",
        "",
    ]
    lines.extend(
        [f"- `{key}`" for key in sorted(result["missing"])] or ["- None"]
    )
    lines.extend(["", "## Unused entries", ""])
    lines.extend(
        [f"- `{key}`" for key in sorted(result["unused"])] or ["- None"]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=Path("paper/phaseguard_paper_draft.md"),
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=Path("paper/references.bib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/citation_audit.md"),
    )
    parser.add_argument(
        "--require-all-used",
        action="store_true",
        help="Also fail if a bibliography entry is unused.",
    )
    args = parser.parse_args()

    result = audit(
        args.manuscript.read_text(encoding="utf-8"),
        args.bibliography.read_text(encoding="utf-8"),
    )
    report = render(result, args.manuscript, args.bibliography)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")

    if result["missing"]:
        return 1
    if args.require_all_used and result["unused"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
