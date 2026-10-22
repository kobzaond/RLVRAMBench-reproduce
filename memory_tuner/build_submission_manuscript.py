#!/usr/bin/env python3
"""Build editable journal-submission files from the canonical manuscript."""

from __future__ import annotations

import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from memory_tuner.check_paper_citations import citation_keys


ABSTRACT_WORD_LIMIT = 250
AI_DECLARATION_HEADING = (
    "## Declaration of Generative AI and AI-assisted technologies "
    "in the writing process"
)
WRITING_AI_DECLARATION_FRAGMENTS = (
    "OpenAI Codex was used to assist with coding and to refine and polish",
    "The authors are responsible for the code, analyses, and content",
)
# One complete declaration covers both coding and text polishing. Requiring a
# second methods declaration duplicated that information in the article.
GENERATED_DISCLOSURE_FRAGMENTS = WRITING_AI_DECLARATION_FRAGMENTS
FIGURE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
ARCHIVAL_REFERENCE_REQUIREMENTS = {
    "hu2022lora": "International Conference on Learning Representations",
    "sheng2024hybridflow": (
        "Proceedings of the Twentieth European Conference on Computer Systems"
    ),
    "zhong2024rlhfuse": (
        "22nd USENIX Symposium on Networked Systems Design and Implementation"
    ),
    "hu2024openrlhf": (
        "Proceedings of the 2025 Conference on Empirical Methods in Natural "
        "Language Processing: System Demonstrations"
    ),
    "mei2024realhf": "Proceedings of Machine Learning and Systems",
    "fu2025areal": "Advances in Neural Information Processing Systems",
    "he2026hetrl": "Proceedings of Machine Learning and Systems",
    "kwon2023pagedattention": (
        "Proceedings of the 29th Symposium on Operating Systems Principles"
    ),
    "zhou2024rlhfmemory": "Machine Learning for Systems Workshop",
    "henderson2018matters": (
        "Proceedings of the Thirty-Second AAAI Conference on Artificial "
        "Intelligence"
    ),
    "agarwal2021precipice": (
        "Advances in Neural Information Processing Systems"
    ),
    "wu2026weave": (
        "20th USENIX Symposium on Operating Systems Design and Implementation"
    ),
    "wu2026rlboost": (
        "23rd USENIX Symposium on Networked Systems Design and Implementation"
    ),
    "wang2026dynarl": (
        "20th USENIX Symposium on Operating Systems Design and Implementation"
    ),
    "tan2026rollart": (
        "20th USENIX Symposium on Operating Systems Design and Implementation"
    ),
    "chen2026robustrl": (
        "20th USENIX Symposium on Operating Systems Design and Implementation"
    ),
    "wang2025sirius": "2025 USENIX Annual Technical Conference",
}


def split_title(text: str) -> tuple[str, str]:
    first, separator, remainder = text.partition("\n")
    if not separator or not first.startswith("# "):
        raise ValueError("manuscript must begin with one level-one title")
    return first[2:].strip(), remainder.lstrip()


def abstract_word_count(text: str) -> int:
    match = re.search(
        r"^## Abstract\s*$\n(.*?)(?=^##\s)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError("manuscript must contain a level-two Abstract")
    abstract = match.group(1).split("**Keywords:**", 1)[0]
    abstract = re.sub(r"<!--.*?-->", " ", abstract, flags=re.DOTALL)
    return len(re.findall(r"\b[\w][\w’'/-]*\b", abstract))


def validate_manuscript(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    normalized = _normalized_text(text)
    count = abstract_word_count(text)
    if count > ABSTRACT_WORD_LIMIT:
        raise ValueError(
            f"abstract has {count} words; limit is {ABSTRACT_WORD_LIMIT}"
        )
    if AI_DECLARATION_HEADING not in text:
        raise ValueError(
            "manuscript must contain the generative-AI declaration"
        )
    for fragment in WRITING_AI_DECLARATION_FRAGMENTS:
        if fragment not in normalized:
            raise ValueError(
                "manuscript must contain the coding/text-polishing "
                f"AI declaration fragment {fragment!r}"
            )


def validate_highlights(path: Path) -> None:
    highlights = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 3 <= len(highlights) <= 5:
        raise ValueError("highlights must contain three to five nonempty lines")
    unbulleted = [
        line
        for line in highlights
        if not (line.startswith("- ") or line.startswith("• "))
    ]
    if unbulleted:
        raise ValueError(
            "each highlight must start with '- ' or '• ': "
            + "; ".join(unbulleted)
        )
    too_long = [line for line in highlights if len(line) > 85]
    if too_long:
        raise ValueError(
            "highlights exceed 85 characters: " + "; ".join(too_long)
        )


def local_figure_paths(manuscript: Path) -> list[Path]:
    text = manuscript.read_text(encoding="utf-8")
    paths = []
    for target in FIGURE_PATTERN.findall(text):
        if "://" in target or target.startswith("#"):
            continue
        paths.append((manuscript.parent / target).resolve())
    return paths


def validate_figure_resources(manuscript: Path) -> int:
    figures = local_figure_paths(manuscript)
    if not figures:
        raise ValueError("manuscript must reference at least one local figure")
    missing = sorted({str(path) for path in figures if not path.is_file()})
    if missing:
        raise ValueError(
            "manuscript references missing figure resources: "
            + "; ".join(missing)
        )
    return len(set(figures))


def validate_table_captions(manuscript: Path) -> int:
    text = manuscript.read_text(encoding="utf-8")
    tables = len(re.findall(r"^\|[-: ]+\|[-:| ]+$", text, flags=re.MULTILINE))
    captions = [int(n) for n in re.findall(
        r"^Table: Table (\d+)\. ", text, flags=re.MULTILINE)]
    if captions != list(range(1, tables + 1)):
        raise ValueError("every table requires one consecutively numbered caption")
    return tables


def validate_generated_tables(manuscript: Path, docx: Path, tex: Path) -> None:
    expected = validate_table_captions(manuscript)
    tex_count = tex.read_text(encoding="utf-8").count(r"\begin{longtable}")
    with zipfile.ZipFile(docx) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    docx_count = sum(node.tag.endswith("}tbl") for node in document.iter())
    captions = sum(node.tag.endswith("}tblCaption") for node in document.iter())
    if (tex_count, docx_count, captions) != (expected, expected, expected):
        raise ValueError(
            f"generated table/caption mismatch: expected {expected}; "
            f"TeX {tex_count}, DOCX tables {docx_count}, captions {captions}")


def validate_generated_figure_counts(
    manuscript: Path,
    docx: Path,
    tex: Path,
) -> None:
    expected = validate_figure_resources(manuscript)
    tex_count = tex.read_text(encoding="utf-8").count(r"\includegraphics")
    with zipfile.ZipFile(docx) as archive:
        docx_count = sum(
            name.startswith("word/media/") and not name.endswith("/")
            for name in archive.namelist()
        )
    if tex_count != expected or docx_count != expected:
        raise ValueError(
            "generated submission omitted figures: "
            f"expected {expected}, TeX contains {tex_count}, "
            f"DOCX contains {docx_count}"
        )


def validate_generated_reference_sets(
    manuscript: Path,
    docx: Path,
    tex: Path,
) -> None:
    expected = citation_keys(manuscript.read_text(encoding="utf-8"))
    tex_keys = set(
        re.findall(
            r"\\bibitem(?:\[[^\]]*\])?\{ref-([^}]+)\}",
            tex.read_text(encoding="utf-8"),
        )
    )
    with zipfile.ZipFile(docx) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    docx_keys = set(re.findall(r"ref-([A-Za-z0-9:_-]+)", document_xml))
    if tex_keys != expected or docx_keys != expected:
        raise ValueError(
            "generated submission citation set mismatch: "
            f"expected {len(expected)}, TeX contains {len(tex_keys)}, "
            f"DOCX contains {len(docx_keys)}; "
            f"TeX missing {sorted(expected - tex_keys)}, "
            f"DOCX missing {sorted(expected - docx_keys)}"
        )


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _docx_document_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    paragraphs = []
    for paragraph in root.iter():
        if not paragraph.tag.endswith("}p"):
            continue
        text = "".join(
            node.text or ""
            for node in paragraph.iter()
            if node.tag.endswith("}t")
        )
        if text:
            paragraphs.append(text)
    return _normalized_text(" ".join(paragraphs))


def validate_generated_disclosures(docx: Path, tex: Path) -> None:
    outputs = {
        "TeX": _normalized_text(tex.read_text(encoding="utf-8")),
        "DOCX": _docx_document_text(docx),
    }
    errors = []
    for label, text in outputs.items():
        for fragment in GENERATED_DISCLOSURE_FRAGMENTS:
            if fragment not in text:
                errors.append(
                    f"{label} lacks required disclosure fragment {fragment!r}"
                )
    if errors:
        raise ValueError(
            "generated disclosure validation failed: " + "; ".join(errors)
        )


def _tex_reference_blocks(text: str) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"\\bibitem(?:\[[^\]]*\])?\{ref-([^}]+)\}",
            text,
        )
    )
    return {
        match.group(1): _normalized_text(
            text[
                match.start() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                )
            ]
        )
        for index, match in enumerate(matches)
    }


def _docx_reference_blocks(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    blocks = {}

    def attribute(node: ET.Element, suffix: str) -> str:
        return next(
            (
                value
                for name, value in node.attrib.items()
                if name.endswith(f"}}{suffix}") or name == suffix
            ),
            "",
        )

    for parent in root.iter():
        active: dict[str, str] = {}
        for child in list(parent):
            if child.tag.endswith("}bookmarkStart"):
                name = attribute(child, "name")
                if name.startswith("ref-"):
                    active[attribute(child, "id")] = name[4:]
                continue
            if child.tag.endswith("}bookmarkEnd"):
                active.pop(attribute(child, "id"), None)
                continue
            if not child.tag.endswith("}p"):
                continue
            keys = list(active.values())
            for node in child.iter():
                if not node.tag.endswith("}bookmarkStart"):
                    continue
                name = attribute(node, "name")
                if name.startswith("ref-"):
                    keys.append(name[4:])
            text = _normalized_text(" ".join(child.itertext()))
            for key in keys:
                blocks[key] = text
    return blocks


def validate_generated_archival_references(
    docx: Path,
    tex: Path,
    requirements: Mapping[str, str] = ARCHIVAL_REFERENCE_REQUIREMENTS,
) -> None:
    tex_blocks = _tex_reference_blocks(tex.read_text(encoding="utf-8"))
    docx_blocks = _docx_reference_blocks(docx)
    errors = []
    for key, venue in requirements.items():
        for label, blocks in (("TeX", tex_blocks), ("DOCX", docx_blocks)):
            block = blocks.get(key)
            if block is None:
                errors.append(f"{label} lacks rendered reference {key}")
                continue
            folded = block.casefold()
            if venue.casefold() not in folded:
                errors.append(
                    f"{label} reference {key} lacks archival venue {venue!r}"
                )
            if "arxiv" in folded:
                errors.append(
                    f"{label} reference {key} retains an arXiv fallback"
                )
    if errors:
        raise ValueError(
            "generated archival-reference validation failed: "
            + "; ".join(errors)
        )


def build(
    manuscript: Path,
    bibliography: Path,
    output_dir: Path,
    highlights: Path,
) -> tuple[Path, Path]:
    validate_manuscript(manuscript)
    validate_highlights(highlights)
    validate_figure_resources(manuscript)
    import pypandoc

    output_dir.mkdir(parents=True, exist_ok=True)
    title, body = split_title(manuscript.read_text(encoding="utf-8"))
    common = [
        "--standalone",
        "--citeproc",
        f"--bibliography={bibliography.resolve()}",
        f"--resource-path={manuscript.parent.resolve()}",
        f"--metadata=title:{title}",
        "--metadata=reference-section-title:References",
    ]
    docx = output_dir / "RLVRAMBench_JSS_submission.docx"
    tex = output_dir / "RLVRAMBench_JSS_submission.tex"
    pypandoc.convert_text(
        body,
        "docx",
        format="markdown",
        outputfile=str(docx),
        extra_args=common,
    )
    latex_body = re.sub(r"!\[Figure [1-9][0-9]*\. ", "![", body)
    latex_body = re.sub(r"^Table: Table [1-9][0-9]*\. ", "Table: ", latex_body,
                        flags=re.MULTILINE)
    for target in FIGURE_PATTERN.findall(latex_body):
        if "://" in target or target.startswith("#"):
            continue
        source = (manuscript.parent / target).resolve()
        vector = source.with_suffix(".pdf")
        if vector.is_file():
            source = vector
        relative = Path(os.path.relpath(source, output_dir.resolve())).as_posix()
        latex_body = latex_body.replace(f"]({target})", f"]({relative})")
    pypandoc.convert_text(
        latex_body,
        "latex",
        format="markdown",
        outputfile=str(tex),
        extra_args=[*common, "--wrap=none", "--variable=papersize:a4",
                    "--variable=geometry:margin=0.8in", "--variable=fontsize:11pt"],
    )
    validate_generated_figure_counts(manuscript, docx, tex)
    validate_generated_tables(manuscript, docx, tex)
    validate_generated_reference_sets(manuscript, docx, tex)
    validate_generated_archival_references(docx, tex)
    validate_generated_disclosures(docx, tex)
    return docx, tex


def main() -> None:
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
        "--output-dir",
        type=Path,
        default=Path("paper/submission"),
    )
    parser.add_argument(
        "--highlights",
        type=Path,
        default=Path("paper/submission/highlights.txt"),
    )
    args = parser.parse_args()
    docx, tex = build(
        args.manuscript,
        args.bibliography,
        args.output_dir,
        args.highlights,
    )
    print(f"wrote {docx} and {tex}")


if __name__ == "__main__":
    main()
