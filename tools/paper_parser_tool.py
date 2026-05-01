from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SECTION_PATTERNS = {
    "abstract": re.compile(r"^\s*abstract\s*$", re.IGNORECASE),
    "introduction": re.compile(r"^\s*(\d+\.?\s*)?introduction\s*$", re.IGNORECASE),
    "literature_review": re.compile(
        r"^\s*(\d+\.?\s*)?(related work|literature review|background)\s*$",
        re.IGNORECASE,
    ),
    "methodology": re.compile(
        r"^\s*(\d+\.?\s*)?(methodology|methods?|materials and methods?)\s*$",
        re.IGNORECASE,
    ),
    "results": re.compile(r"^\s*(\d+\.?\s*)?(results?|evaluation)\s*$", re.IGNORECASE),
    "discussion": re.compile(r"^\s*(\d+\.?\s*)?discussion\s*$", re.IGNORECASE),
    "conclusion": re.compile(
        r"^\s*(\d+\.?\s*)?(conclusion|conclusions|future work)\s*$",
        re.IGNORECASE,
    ),
    "references": re.compile(
        r"^\s*(references|bibliography|works cited)\s*$",
        re.IGNORECASE,
    ),
}

HEADING_PREFIX = r"(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+)|(?:[A-Z]))[.)]?\s*"
SECTION_HEADER_PATTERNS = {
    "abstract": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?abstract\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "introduction": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?introduction\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "literature_review": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?(related work|literature review|background)\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "methodology": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?(methodology|method|methods|materials and methods|approach|experimental setup)\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "results": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?(results|result|evaluation|experiments)\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "discussion": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?(discussion|analysis|limitations)\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "conclusion": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?(conclusion|conclusions|future work|summary)\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
    "references": re.compile(
        rf"^\s*(?:{HEADING_PREFIX})?(references|bibliography|works cited)\b\s*[:\-.]?\s*(.*)$",
        re.IGNORECASE,
    ),
}

INLINE_SECTION_MARKER = re.compile(
    r"\b(?:[IVXLC]{1,6}[.)]|\d+(?:\.\d+)*[.)]?)\s+(ABSTRACT|INTRODUCTION|BACKGROUND|RELATED\s+WORK|LITERATURE\s+REVIEW|METHODS?|METHODOLOGY|SYSTEM\s+ARCHITECTURE|EXPERIMENTS?|RESULTS?|DISCUSSION|CONCLUSIONS?|FUTURE\s+WORK|REFERENCES|BIBLIOGRAPHY|WORKS\s+CITED)\b",
    re.IGNORECASE,
)

ABSTRACT_BOUNDARY_MARKER = re.compile(
    r"\b(?:index\s+terms|keywords?|(?:[IVXLC]{1,6}[.)]|\d+(?:\.\d+)*[.)]?)\s+(introduction|background|related\s+work|literature\s+review|methods?|methodology))\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r", "\n")).strip()


def _clean_extracted_text(text: str) -> str:
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = text.replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _join_line_words(words: List[Dict[str, Any]]) -> str:
    ordered = sorted(words, key=lambda word: float(word["x0"]))
    text = " ".join(str(word["text"]) for word in ordered)
    return re.sub(r"\s+([,.;:!?\)])", r"\1", text).strip()


def _words_to_lines(words: List[Dict[str, Any]], y_tolerance: float = 3.0) -> List[Tuple[float, str]]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: (float(word["top"]), float(word["x0"])))
    groups: List[Tuple[float, List[Dict[str, Any]]]] = []
    for word in sorted_words:
        top = float(word["top"])
        if not groups:
            groups.append((top, [word]))
            continue

        prev_top, prev_words = groups[-1]
        if abs(top - prev_top) <= y_tolerance:
            prev_words.append(word)
        else:
            groups.append((top, [word]))

    lines: List[Tuple[float, str]] = []
    for top, line_words in groups:
        text = _join_line_words(line_words)
        if text:
            lines.append((top, text))
    return lines


def _extract_page_text(page: Any) -> str:
    return page.extract_text(x_tolerance=2, y_tolerance=3) or ""


def _split_embedded_header_line(line: str) -> List[str]:
    remaining = line.strip()
    if not remaining:
        return []

    pieces: List[str] = []
    while remaining:
        match = INLINE_SECTION_MARKER.search(remaining)
        if not match:
            pieces.append(remaining.strip())
            break

        split_index: Optional[int] = None
        if match.start() >= 20:
            split_index = match.start()
        elif match.start() == 0:
            for later_match in INLINE_SECTION_MARKER.finditer(remaining, 1):
                if later_match.start() >= 12:
                    split_index = later_match.start()
                    break

        if split_index is None:
            pieces.append(remaining.strip())
            break

        before = remaining[:split_index].strip()
        if before:
            pieces.append(before)
        remaining = remaining[split_index:].strip()

    return [piece for piece in pieces if piece]


def _trim_abstract_at_boundary(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""

    match = ABSTRACT_BOUNDARY_MARKER.search(cleaned)
    if match and match.start() > 80:
        return normalize_whitespace(cleaned[: match.start()])
    return cleaned


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "pymupdf is required to extract text from PDFs. Install with: pip install pymupdf"
        ) from exc

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    pages: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            # get_text("blocks") identifies columns natively and sorts by natural reading order
            blocks = page.get_text("blocks")
            text_blocks = [b[4].strip() for b in blocks if len(b) > 4 and b[4].strip()]
            pages.append("\n\n".join(text_blocks))

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")

    return "\n\n".join(pages)


def extract_title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Untitled Paper"

    for line in lines[:5]:
        if len(line.split()) >= 4 and not SECTION_PATTERNS["abstract"].match(line):
            return line
    return lines[0]


def _match_section_header(line: str) -> Tuple[Optional[str], str]:
    for section_name, pattern in SECTION_HEADER_PATTERNS.items():
        match = pattern.match(line)
        if not match:
            continue

        if section_name == "literature_review":
            inline_text = match.group(2) or ""
        elif match.lastindex and match.lastindex >= 2:
            inline_text = match.group(2) or ""
        elif match.lastindex and match.lastindex >= 1:
            inline_text = match.group(1) or ""
        else:
            inline_text = ""
        return section_name, normalize_whitespace(inline_text)

    return None, ""


def split_into_sections(text: str) -> Dict[str, str]:
    expanded_lines: List[str] = []
    for raw_line in text.splitlines():
        expanded_lines.extend(_split_embedded_header_line(raw_line))

    lines = [line.strip() for line in expanded_lines if line.strip()]
    sections: Dict[str, List[str]] = {}
    current_section = "preamble"
    sections[current_section] = []

    for line in lines:
        matched_section, inline_text = _match_section_header(line)
        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
            if inline_text:
                sections[current_section].append(inline_text)
            continue

        sections.setdefault(current_section, []).append(line)

    return {
        name: normalize_whitespace("\n".join(content))
        for name, content in sections.items()
        if content
    }


def extract_abstract(sections: Dict[str, str], text: str) -> str:
    if "abstract" in sections:
        return _trim_abstract_at_boundary(sections["abstract"])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    abstract_lines: List[str] = []
    in_abstract = False
    for line in lines:
        if SECTION_PATTERNS["abstract"].match(line):
            in_abstract = True
            continue
        if in_abstract and any(pattern.match(line) for pattern in SECTION_PATTERNS.values()):
            break
        if in_abstract:
            abstract_lines.append(line)

    return _trim_abstract_at_boundary("\n".join(abstract_lines))


def extract_bibliography(sections: Dict[str, str], text: str) -> List[str]:
    references_text = sections.get("references", "")
    if not references_text:
        lines = text.splitlines()
        start_index = -1
        for idx, line in enumerate(lines):
            matched_section, _inline = _match_section_header(line.strip())
            if matched_section == "references":
                start_index = idx + 1
                break
        if start_index >= 0:
            references_text = "\n".join(lines[start_index:])

    if not references_text:
        return []

    lines = [line.strip() for line in references_text.splitlines() if line.strip()]
    bibliography: List[str] = []
    current_entry = ""
    marker_pattern = re.compile(r"^(\[\d+\]|\d+[.)])\s+")

    for line in lines:
        if marker_pattern.match(line):
            if current_entry:
                bibliography.append(normalize_whitespace(current_entry))
            current_entry = marker_pattern.sub("", line, count=1)
        elif current_entry:
            current_entry = f"{current_entry} {line}".strip()
        else:
            bibliography.append(normalize_whitespace(line))

    if current_entry:
        bibliography.append(normalize_whitespace(current_entry))

    deduplicated: List[str] = []
    seen = set()
    for entry in bibliography:
        if entry and entry not in seen:
            deduplicated.append(entry)
            seen.add(entry)

    return deduplicated[:25]


def extract_key_findings(sections: Dict[str, str]) -> List[str]:
    findings_source = (
        sections.get("results", "")
        or sections.get("discussion", "")
        or sections.get("conclusion", "")
    )
    if not findings_source:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", findings_source)
    cleaned = [normalize_whitespace(sentence) for sentence in sentences if sentence.strip()]
    filtered = [
        sentence
        for sentence in cleaned
        if len(sentence.split()) >= 4
        and not sentence.lower().startswith(("table", "fig", "index terms"))
    ]
    return filtered[:5]


def extract_authors(text: str, title: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or title == "Untitled Paper":
        return ""

    author_lines = []
    found_title = False
    for line in lines[:30]:
        if not found_title:
            if line in title or title in line:
                found_title = True
            continue

        if SECTION_PATTERNS["abstract"].match(line) or "abstract" in line.lower():
            break

        author_lines.append(line)

    return "\n".join(author_lines).strip()


def parse_paper(pdf_path: str) -> Dict[str, Any]:
    text = extract_text_from_pdf(pdf_path)
    sections = split_into_sections(text)
    title = extract_title(text)
    authors = extract_authors(text, title)
    return {
        "pdf_path": pdf_path,
        "title": title,
        "authors": authors,
        "abstract": extract_abstract(sections, text),
        "sections": sections,
        "references": extract_bibliography(sections, text),
        "key_findings": extract_key_findings(sections),
    }


def parse_multiple_papers(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    return [parse_paper(pdf_path) for pdf_path in pdf_paths]
