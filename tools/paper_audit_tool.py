from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# The sections every academic paper must have
REQUIRED_SECTIONS: List[str] = [
    "Abstract",
    "Introduction",
    "Methodology",
    "Conclusion",
    "References",
]

# Aliases — different papers write section names differently
SECTION_ALIASES: Dict[str, List[str]] = {
    "Abstract":      ["abstract", "summary", "overview"],
    "Introduction":  ["introduction", "background", "motivation", "1. introduction", "i. introduction"],
    "Methodology":   ["methodology", "methods", "approach", "proposed method", "system design",
                      "3. methodology", "ii. methodology", "proposed approach"],
    "Conclusion":    ["conclusion", "conclusions", "concluding remarks", "summary and conclusion",
                      "5. conclusion", "6. conclusion", "v. conclusion"],
    "References":    ["references", "bibliography", "works cited"],
}


def paper_audit_tool(
    own_paper_path: str,
    paper_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Audit a student's academic paper for quality, completeness, and originality.

    Performs three independent checks:
      1. Section presence check  — verifies all required sections exist.
      2. TF-IDF plagiarism check — cosine similarity against reference papers.
      3. Citation consistency    — cross-references in-text citations vs reference list.

    Then combines the three results into a single integrity_score (0.0 – 1.0).

    Args:
        own_paper_path:  Absolute or relative path to the student's paper PDF.
        paper_summaries: List of dicts from Member 2's literature review agent.
                         Each dict is expected to have a "raw_text" or "abstract" key.

    Returns:
        Dictionary with keys:
            missing_sections   (List[str]):  Sections not found in the paper.
            formatting_issues  (List[str]):  Detected formatting problems.
            plagiarism_score   (float):      0.0 (original) – 1.0 (identical).
            citation_issues    (List[str]):  Citations that are inconsistent.
            integrity_score    (float):      Overall quality score 0.0 – 1.0.
            raw_text           (str):        Full extracted text for LLM use.

    Raises:
        FileNotFoundError: If own_paper_path does not exist.
        ValueError:        If the PDF yields no extractable text.
    """
    path = Path(own_paper_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper PDF not found: {own_paper_path}")

    # ------------------------------------------------------------------ #
    # Step 1 — Extract text from the student's PDF
    # ------------------------------------------------------------------ #
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages: List[str] = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF '{own_paper_path}': {exc}") from exc

    full_text: str = "\n".join(pages).strip()
    if not full_text:
        raise ValueError(
            f"PDF '{own_paper_path}' yielded no extractable text. "
            "Make sure the file is text-based, not a scanned image."
        )

    text_lower: str = full_text.lower()

    # ------------------------------------------------------------------ #
    # Step 2 — Section presence check
    # ------------------------------------------------------------------ #
    missing_sections, formatting_issues = _check_sections(text_lower)

    # ------------------------------------------------------------------ #
    # Step 3 — Plagiarism check via TF-IDF cosine similarity
    # ------------------------------------------------------------------ #
    plagiarism_score: float = _compute_plagiarism_score(full_text, paper_summaries)

    # ------------------------------------------------------------------ #
    # Step 4 — Citation consistency check
    # ------------------------------------------------------------------ #
    citation_issues: List[str] = _check_citations(full_text)

    # ------------------------------------------------------------------ #
    # Step 5 — Compute composite integrity_score
    # ------------------------------------------------------------------ #
    integrity_score: float = _compute_integrity_score(
        missing_sections, plagiarism_score, citation_issues, full_text
    )

    return {
        "missing_sections":  missing_sections,
        "formatting_issues": formatting_issues,
        "plagiarism_score":  round(plagiarism_score, 4),
        "citation_issues":   citation_issues,
        "integrity_score":   round(integrity_score, 4),
        "raw_text":          full_text,
    }


# ====================================================================== #
# Private helpers
# ====================================================================== #

def _check_sections(text_lower: str) -> Tuple[List[str], List[str]]:
    """
    Detect which required sections are missing and note formatting problems.

    Args:
        text_lower: Lowercase full text of the paper.

    Returns:
        Tuple of (missing_sections, formatting_issues).
    """
    missing: List[str] = []
    issues: List[str] = []

    for section, aliases in SECTION_ALIASES.items():
        found = any(alias in text_lower for alias in aliases)
        if not found:
            missing.append(section)

    # Formatting checks
    if len(text_lower) < 1000:
        issues.append("Paper is very short (under 1000 characters) — may be incomplete.")

    if "fig." not in text_lower and "figure" not in text_lower and "table" not in text_lower:
        issues.append("No figures or tables detected — consider adding visual aids.")

    if text_lower.count("\n\n") < 5:
        issues.append("Very few paragraph breaks detected — check formatting.")

    return missing, issues


def _compute_plagiarism_score(
    own_text: str,
    paper_summaries: List[Dict[str, Any]],
) -> float:
    """
    Compute the maximum TF-IDF cosine similarity between the student's paper
    and any of the reference papers provided by Member 2.

    A score of 0.0 means completely original; 1.0 means identical.

    Args:
        own_text:        Full text of the student's paper.
        paper_summaries: List of summary dicts from the literature review agent.

    Returns:
        Float in range [0.0, 1.0].
    """
    if not paper_summaries:
        return 0.0

    # Collect reference texts — try raw_text first, fall back to abstract
    reference_texts: List[str] = []
    for summary in paper_summaries:
        text = (
            summary.get("raw_text")
            or summary.get("abstract")
            or summary.get("key_findings", "")
            or ""
        )
        if isinstance(text, str) and text.strip():
            reference_texts.append(text.strip())

    if not reference_texts:
        return 0.0

    try:
        corpus = [own_text] + reference_texts
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),   # unigrams + bigrams for better sensitivity
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # Row 0 = own paper; rows 1+ = reference papers
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return float(similarities.max())
    except Exception:
        return 0.0


def _check_citations(full_text: str) -> List[str]:
    """
    Cross-reference in-text citations against the reference list.

    Supports two common citation styles:
      - Numbered:  [1], [2], [1,2], [1-3]
      - Author-year: (Smith, 2021), (Jones et al., 2020)

    Args:
        full_text: Full extracted text of the paper.

    Returns:
        List of issue strings describing inconsistencies found.
    """
    issues: List[str] = []
    text_lower = full_text.lower()

    # Find the references section boundary
    ref_section_start = _find_references_start(text_lower)
    body_text = full_text[:ref_section_start] if ref_section_start else full_text
    ref_text = full_text[ref_section_start:] if ref_section_start else ""

    # --- Numbered citations ---
    intext_numbers = set(
        num
        for match in re.findall(r"\[(\d+(?:[,\s]\d+)*)\]", body_text)
        for num in re.findall(r"\d+", match)
    )
    ref_numbers = set(re.findall(r"^\s*\[(\d+)\]", ref_text, re.MULTILINE))

    for num in sorted(intext_numbers, key=int):
        if num not in ref_numbers:
            issues.append(f"In-text citation [{num}] has no matching reference entry.")

    # --- Author-year citations ---
    intext_authors = set(
        match.split(",")[0].strip().lower()
        for match in re.findall(r"\(([A-Z][a-z]+(?:\s+et\s+al\.)?),\s*\d{4}\)", body_text)
    )
    for author in intext_authors:
        # Check if the author's last name appears anywhere in the references section
        if ref_text and author not in ref_text.lower():
            issues.append(
                f"Author-year citation '({author.title()}, ...)' not found in References."
            )

    if not issues:
        if not intext_numbers and not intext_authors:
            issues.append("No citations detected in the paper body — citations are required.")

    return issues


def _find_references_start(text_lower: str) -> int:
    """
    Return the character index where the references section starts.

    Args:
        text_lower: Lowercase full text.

    Returns:
        Index of references section start, or 0 if not found.
    """
    for pattern in ["\nreferences\n", "\nbibliography\n", "\nworks cited\n"]:
        idx = text_lower.rfind(pattern)   # rfind = last occurrence (true refs section)
        if idx != -1:
            return idx
    return 0


def _compute_integrity_score(
    missing_sections: List[str],
    plagiarism_score: float,
    citation_issues: List[str],
    full_text: str,
) -> float:
    """
    Combine three audit signals into one integrity score from 0.0 to 1.0.

    Scoring breakdown:
      - Start at 1.0
      - Deduct 0.10 per missing required section  (max -0.50)
      - Deduct plagiarism_score * 0.40            (max -0.40)
      - Deduct 0.05 per citation issue            (max -0.20)
      - Small length bonus if paper is substantial

    Args:
        missing_sections: Sections not found.
        plagiarism_score: TF-IDF similarity score.
        citation_issues:  List of citation problems.
        full_text:        Full paper text (used for length bonus).

    Returns:
        Float clamped to [0.0, 1.0].
    """
    score = 1.0

    # Deduct for missing sections
    score -= min(len(missing_sections) * 0.10, 0.50)

    # Deduct for plagiarism
    score -= plagiarism_score * 0.40

    # Deduct for citation issues
    score -= min(len(citation_issues) * 0.05, 0.20)

    # Small bonus for length (a real paper has substantial content)
    if len(full_text) > 5000:
        score = min(score + 0.05, 1.0)

    return round(max(score, 0.0), 4)