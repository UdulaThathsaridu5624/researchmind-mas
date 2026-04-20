from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber


def proposal_analyzer_tool(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze a research proposal or academic paper PDF and extract structured metadata.

    Works with both standard proposal format (Keywords/Objectives/Scope/Methodology)
    and IEEE conference paper format (Index Terms / numbered contributions /
    System Architecture / Module Design / etc.).

    Reads all pages via pdfplumber then applies layered regex heuristics. If a
    section cannot be found, an empty string or list is returned so the calling
    agent can compensate using the always-present raw_text field.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Dictionary with keys:
            title       (str):       Paper or proposal title.
            objectives  (List[str]): Extracted research objectives or contributions.
            scope       (str):       Scope of the research.
            methodology (str):       Methodology / system architecture description.
            keywords    (List[str]): Extracted keywords or index terms.
            raw_text    (str):       Full concatenated text for LLM use.

    Raises:
        FileNotFoundError: If pdf_path does not point to an existing file.
        ValueError: If the PDF yields no extractable text (e.g. scanned image).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Proposal PDF not found: {pdf_path}")

    try:
        with pdfplumber.open(str(path)) as pdf:
            pages_text_default: List[str] = [page.extract_text() or "" for page in pdf.pages]
            pages_text_column: List[str] = [_extract_page_text(page) for page in pdf.pages]
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF '{pdf_path}': {exc}") from exc

    full_text_default = _postprocess_raw_text("\n".join(pages_text_default).strip())
    full_text_column = _postprocess_raw_text("\n".join(pages_text_column).strip())

    full_text: str = full_text_default
    # Rejoin hyphenated line-breaks common in two-column PDF layouts.
    # Example: "code gen-\neration" becomes "code generation".
    if not full_text:
        raise ValueError(
            f"PDF '{pdf_path}' yielded no extractable text. "
            "Ensure the file is text-based, not a scanned image."
        )

    objectives = _extract_objectives(full_text_default)
    fallback_objectives = _extract_objectives(full_text_column)
    if _objective_score(fallback_objectives) > _objective_score(objectives):
        objectives = fallback_objectives

    scope = _extract_scope(full_text_default) or _extract_scope(full_text_column)
    methodology = _extract_methodology(full_text_default) or _extract_methodology(full_text_column)

    keywords = _extract_keywords(full_text_default)
    fallback_keywords = _extract_keywords(full_text_column)
    if len(fallback_keywords) > len(keywords):
        keywords = fallback_keywords

    return {
        "title": _extract_title(pages_text_default),
        "objectives": objectives,
        "scope": scope,
        "methodology": methodology,
        "keywords": keywords,
        "raw_text": full_text,
    }


def _extract_title(pages_text: List[str]) -> str:
    """
    Return the most likely title from the first page.

    Skips very short lines (author names, affiliations) and very long ones
    (body paragraphs). Returns the first qualifying line.
    """
    first_page = pages_text[0] if pages_text else ""
    for line in first_page.splitlines():
        line = line.strip()
        if 10 <= len(line) <= 200:
            return line
    return "Unknown Title"


def _extract_keywords(text: str) -> List[str]:
    """
    Extract keywords / index terms from multiple common formats.

    Supported formats:
    - Proposal format  : ``Keywords: word1, word2``
    - IEEE format      : ``Index Terms-word1, word2``
    - Hyphen variant   : ``Index Terms - word1, word2``
    - Colon variant    : ``Index Terms: word1, word2``

    Two-column PDFs often bleed body sentences into the keyword list.
    Each candidate keyword is validated: if it contains a period or looks
    like a sentence fragment (>8 words), it and all subsequent candidates
    are discarded.
    """
    pattern = (
        r"(?:keywords?|index\s+terms?)"
        r"\s*[:\-\u2013\u2014]\s*"
        r"(.+?)"
        r"(?:\n\n|\n[A-Z\d]|\Z)"
    )
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    raw = match.group(1).replace("\n", " ").strip()
    parts = re.split(r"[,;\u00b7\u2022]\s*", raw)
    keywords: List[str] = []
    seen = set()
    for part in parts:
        part = re.sub(r"\s+", " ", part.strip())
        part = re.sub(r"^[^\w@]+|[^\w\)\]]+$", "", part)
        if not part:
            continue

        word_count = len(part.split())
        if "." in part or word_count > 8:
            continue
        lower = part.lower()
        if lower in seen:
            continue
        keywords.append(part)
        seen.add(lower)
        if len(keywords) >= 12:
            break
    return keywords


def _extract_objectives(text: str) -> List[str]:
    """
    Extract research objectives or contributions using a multi-strategy approach.

    Strategy 1:
        Looks for headings like 'Objectives', 'Research Objectives', 'Aims', 'Goals'.

    Strategy 2:
        Looks for inline numbered lists introduced by phrases like
        'principal contributions:', 'contributions:', or 'this paper presents'.

    Returns a deduplicated list of objective strings.
    """
    contribution_items = _extract_numbered_contributions(text)
    if contribution_items:
        return contribution_items

    section_text = _extract_section(
        text,
        r"(?:research\s+)?(?:objectives?|aims?|goals?)",
        stop_before_references=True,
    )
    if section_text:
        items = re.split(r"\n\s*(?:\d+[.)]\s*|[ivxIVX]+[.)]\s*|[-\u2022*]\s+)", section_text)
        cleaned = [item.replace("\n", " ").strip() for item in items if item.strip()]
        result = [item for item in cleaned if len(item) > 5]
        if result:
            return result

    return []


def _extract_scope(text: str) -> str:
    """
    Extract scope section, trying multiple heading variants.

    Falls back to an empty string if no scope-like section is found.
    """
    return _extract_section(
        text,
        r"scope(?:\s+of\s+(?:the\s+)?(?:research|study|work))?",
        stop_before_references=True,
    )


def _extract_methodology(text: str) -> str:
    """
    Extract methodology / architecture description using an ordered fallback chain.

    Priority order:
    1. 'Methodology' / 'Research Methodology' / 'Methods'
    2. 'System Architecture'
    3. 'Module Design' / 'Module Design and Implementation'
    4. 'Approach' / 'Proposed Approach'
    5. 'Implementation'

    Matches that appear to be reference-list bleed-in are discarded.
    """
    candidates = [
        r"(?:research\s+)?method(?:ology|s)?",
        r"system\s*architecture",
        r"module\s*design(?:\s+and\s+implementation)?",
        r"(?:proposed\s+)?approach",
        r"implementation",
    ]
    for heading in candidates:
        result = _extract_section(text, heading, stop_before_references=True)
        if not result:
            result = _extract_freeform_section(text, heading)
        if not result:
            continue

        starts_with_citation = bool(re.match(r"^\s*\[\d+\]", result))
        citation_density = len(re.findall(r"\[\d+\]", result[:300]))
        is_related_work = bool(re.match(r"^\s*[A-Z][a-z]+ et al\.", result))
        if not starts_with_citation and citation_density <= 5 and not is_related_work:
            return _normalize_section_text(result)
    return ""


def _extract_section(
    text: str,
    heading_pattern: str,
    stop_before_references: bool = False,
) -> str:
    """
    Extract the body of a section identified by a heading regex fragment.

    The match starts at the beginning of a line, optionally preceded by a
    section number like '2.' or 'II.', and captures text until either the next
    section heading, the references section, or end of document.
    """
    # A section heading is a short capitalised phrase (≤ 60 chars) that is NOT
    # the start of a sentence (i.e. does not start with "To ", "The ", "A ", etc.)
    # This prevents numbered list items like "2. To evaluate..." from being
    # treated as the next section heading.
    heading_body = (
        r"(?:[A-Z](?!o |he |n |s ))"  # uppercase but not "To/The/An/As"
        r"[^\n]{2,59}"               # rest of heading, capped at ~60 chars total
    )
    if stop_before_references:
        stop_lookahead = (
            r"(?=\n\s*(?:\d+[\.\s]+|[IVXLC]+[\.\s]+)?"
            + heading_body + r"\n"
            r"|\n\s*(?:references?|bibliography)\s*\n"
            r"|\Z)"
        )
    else:
        stop_lookahead = (
            r"(?=\n\s*(?:\d+[\.\s]+|[IVXLC]+[\.\s]+)?"
            + heading_body + r"\n"
            r"|\Z)"
        )

    pattern = (
        r"(?:^|\n)"
        r"(?:\d+[\.\s]+|[IVXLC]+[\.\s]+)?"
        + heading_pattern
        + r"[^\n]*\n"
        + r"(.*?)"
        + stop_lookahead
    )
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_page_text(page: Any) -> str:
    """
    Extract page text with a column-aware fallback for IEEE-style papers.

    If two columns are detected, text is reconstructed left column first,
    then right column, to reduce cross-column interleaving.
    """
    words = page.extract_words(x_tolerance=2, y_tolerance=2) or []
    if not words:
        return page.extract_text() or ""

    page_width = float(getattr(page, "width", 0) or 0)
    if _is_two_column_layout(words, page_width):
        mid = page_width / 2.0
        gutter = page_width * 0.04
        page_height = float(getattr(page, "height", 0) or 0)
        left_text = (
            page.crop((0.0, 0.0, max(mid - gutter, 1.0), page_height)).extract_text() or ""
        )
        right_text = (
            page.crop((min(mid + gutter, page_width - 1.0), 0.0, page_width, page_height)).extract_text() or ""
        )
        if left_text and right_text:
            return f"{left_text}\n{right_text}"

    return _words_to_text(words)


def _is_two_column_layout(words: List[Dict[str, Any]], page_width: float) -> bool:
    """Heuristic detector for two-column academic page layout."""
    if page_width <= 0 or len(words) < 120:
        return False

    mid = page_width / 2.0
    gutter = page_width * 0.08
    left_count = 0
    right_count = 0
    for word in words:
        center = (float(word.get("x0", 0)) + float(word.get("x1", 0))) / 2.0
        if center <= (mid - gutter):
            left_count += 1
        elif center >= (mid + gutter):
            right_count += 1

    if left_count < 40 or right_count < 40:
        return False
    ratio = left_count / max(right_count, 1)
    return 0.35 <= ratio <= 2.8


def _words_to_text(words: List[Dict[str, Any]]) -> str:
    """Convert pdfplumber word list to line-oriented text."""
    if not words:
        return ""

    sorted_words = sorted(
        words,
        key=lambda w: (round(float(w.get("top", 0)) / 2.0), float(w.get("x0", 0))),
    )

    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = []
    current_top = None
    tolerance = 3.0

    for word in sorted_words:
        top = float(word.get("top", 0))
        if current_top is None or abs(top - current_top) <= tolerance:
            current_line.append(word)
            if current_top is None:
                current_top = top
            else:
                current_top = (current_top + top) / 2.0
        else:
            lines.append(current_line)
            current_line = [word]
            current_top = top

    if current_line:
        lines.append(current_line)

    text_lines: List[str] = []
    for line in lines:
        line_words = sorted(line, key=lambda w: float(w.get("x0", 0)))
        text_lines.append(" ".join(str(w.get("text", "")).strip() for w in line_words).strip())

    return "\n".join(line for line in text_lines if line)


def _extract_numbered_contributions(text: str) -> List[str]:
    """
    Extract numbered contribution/objective lists near contribution cues.

    This handles patterns like:
    - "four principal contributions:"
    - "contributions:"
    followed by "1) ... 2) ... 3) ..."
    """
    cue_pattern = r"(?:principal\s+contributions?|contributions?\s*:|this\s+paper\s+presents?)"
    cue_match = re.search(cue_pattern, text, re.IGNORECASE)
    if not cue_match:
        return []

    window = text[cue_match.start(): cue_match.start() + 3000]
    window = re.split(r"\bindex\s+terms?\b", window, flags=re.IGNORECASE)[0]

    # IEEE papers commonly use inline "(1) ... (2) ... (3) ..." contribution format.
    inline_pattern = r"\(\s*\d+\s*\)\s*(.+?)(?=\(\s*\d+\s*\)|\n\s*[IVXLC]+\.\s|\Z)"
    inline_matches = re.findall(inline_pattern, window, flags=re.IGNORECASE | re.DOTALL)
    inline_cleaned = _clean_contribution_items(inline_matches)
    if len(inline_cleaned) >= 3:
        return inline_cleaned

    line_pattern = r"(?:^|\n)\s*(?:\(?\d+\)?[.)])\s*(.+?)(?=(?:\n\s*\(?\d+\)?[.)])|\n\s*[IVXLC]+\.\s|\Z)"
    line_matches = re.findall(line_pattern, window, flags=re.IGNORECASE | re.DOTALL)
    return _clean_contribution_items(line_matches)


def _extract_freeform_section(text: str, heading_pattern: str) -> str:
    """
    Fallback extractor when strict heading parsing fails due interleaved text.
    """
    match = re.search(heading_pattern, text, re.IGNORECASE)
    if not match:
        return ""

    start = match.end()
    rest = text[start:start + 2200]
    stop = re.search(r"\n\s*(?:[IVXLC]+\.\s+[A-Z][^\n]{2,}|references?\s*$)", rest, re.IGNORECASE | re.MULTILINE)
    if stop:
        rest = rest[:stop.start()]
    return rest.strip()


def _normalize_section_text(value: str) -> str:
    """Normalize whitespace and trim obvious table/figure spillover."""
    normalized = re.sub(r"\s+", " ", value).strip()
    normalized = re.sub(r"\b(?:TABLE|FIG)\s*[IVXLC\d]+\b.*$", "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _postprocess_raw_text(text: str) -> str:
    """Normalize raw PDF text while preserving section boundaries."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _objective_score(items: List[str]) -> int:
    """Score objective extraction quality for default-vs-fallback selection."""
    if not items:
        return 0
    score = len(items) * 100
    for item in items:
        length = len(item.strip())
        if 20 <= length <= 220:
            score += 20
        elif length > 450:
            score -= 30
        if "the " in item.lower():
            score += 2
    return score


def _clean_contribution_items(items: List[str]) -> List[str]:
    """Normalize and filter contribution/objective candidates."""
    cleaned: List[str] = []
    for item in items:
        normalized = _normalize_section_text(item)
        normalized = re.split(r"\b(?:table|fig)\s*[IVXLC\d]+\b", normalized, flags=re.IGNORECASE)[0].strip()
        if len(normalized) < 20:
            continue
        if normalized.lower().startswith(("table", "fig")):
            continue
        if len(normalized) > 360:
            normalized = normalized[:360].rsplit(" ", 1)[0].strip() + "..."
        cleaned.append(normalized)
        if len(cleaned) >= 8:
            break
    return cleaned
