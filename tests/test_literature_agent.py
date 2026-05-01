from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agents.literature_review_agent import literature_review_agent
from tools.paper_parser_tool import (
    extract_bibliography,
    parse_multiple_papers,
    parse_paper,
    split_into_sections,
)


def test_parse_multiple_papers_uses_parser_for_each_path() -> None:
    paths = ["paper-1.pdf", "paper-2.pdf"]
    with patch("tools.paper_parser_tool.parse_paper") as mock_parse:
        mock_parse.side_effect = [
            {"title": "Paper 1"},
            {"title": "Paper 2"},
        ]
        parsed = parse_multiple_papers(paths)

    assert len(parsed) == 2
    assert parsed[0]["title"] == "Paper 1"
    assert parsed[1]["title"] == "Paper 2"


def test_literature_review_agent_populates_expected_state_fields() -> None:
    state = {
        "research_topic": "AI assisted research gap identification",
        "paper_pdf_paths": ["paper-1.pdf", "paper-2.pdf"],
        "agent_logs": [],
        "errors": [],
    }
    parsed_papers = [
        {
            "title": "Paper 1",
            "abstract": "This paper studies transformer support for academic writing.",
            "sections": {
                "introduction": "Academic writing remains time consuming for students.",
                "methodology": "We evaluate a local pipeline using two benchmark datasets.",
                "results": "The local pipeline improves review speed and theme detection.",
                "discussion": "The approach remains sensitive to document quality.",
                "conclusion": "Local review support can improve student productivity.",
                "references": "Ref A\nRef B",
            },
            "references": ["Ref A", "Ref B"],
            "key_findings": [
                "The local pipeline improves review speed.",
                "Theme detection becomes more consistent.",
            ],
        },
        {
            "title": "Paper 2",
            "abstract": "This paper explores automated literature synthesis for reviews.",
            "sections": {
                "introduction": "Literature synthesis often involves repetitive reading.",
                "methodology": "We compare semi structured extraction approaches.",
                "results": "Structured extraction improves traceability of findings.",
                "conclusion": "Traceable summaries help downstream research tasks.",
            },
            "references": ["Ref C"],
            "key_findings": ["Structured extraction improves traceability of findings."],
        },
    ]

    with patch(
        "agents.literature_review_agent.parse_multiple_papers",
        return_value=parsed_papers,
    ):
        updated_state = literature_review_agent(state)

    assert len(updated_state["paper_summaries"]) == 2
    assert "Paper 1" in updated_state["citation_map"]
    assert updated_state["citation_map"]["Paper 1"]["reference_count"] == 2
    assert updated_state["core_themes"]
    assert "Paper 1" in updated_state["section_explanations"]
    assert len(updated_state["agent_logs"]) >= 4


def test_literature_review_agent_handles_missing_paper_paths() -> None:
    state = {
        "research_topic": "Test topic",
        "paper_pdf_paths": [],
        "agent_logs": [],
        "errors": [],
    }

    updated_state = literature_review_agent(state)

    assert updated_state.get("errors")
    assert any("No paper paths" in err for err in updated_state["errors"])


def test_missing_pdf_raises_file_not_found() -> None:
    missing_path = Path("does-not-exist.pdf")
    with patch.object(Path, "exists", return_value=False):
        try:
            parse_multiple_papers([str(missing_path)])
        except FileNotFoundError as exc:
            assert "PDF not found" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError to be raised.")


def test_split_into_sections_supports_numbered_and_inline_headers() -> None:
    text = """
    A Practical Study on Local Research Assistants
    Abstract: We evaluate local-first pipelines for literature analysis.
    1 Introduction
    Students spend significant time in repetitive reading tasks.
    2 Methodology
    We benchmark extraction over two curated datasets.
    3 Results: Accuracy improves for section-level summaries.
    IV. Discussion
    OCR quality still influences downstream confidence.
    5 Conclusion
    Local pipelines are useful for early-stage research support.
    References
    [1] Doe et al. Structured synthesis for research assistants.
    """

    sections = split_into_sections(text)

    assert sections["abstract"].startswith("We evaluate local-first pipelines")
    assert "repetitive reading" in sections["introduction"]
    assert "benchmark extraction" in sections["methodology"]
    assert "Accuracy improves" in sections["results"]
    assert "OCR quality" in sections["discussion"]
    assert "early-stage research support" in sections["conclusion"]


def test_split_into_sections_extracts_literature_review_separately() -> None:
    text = """
    A Study on Local Research Agents
    I. INTRODUCTION
    This paper introduces the overall problem context.
    II. RELATED WORK
    Existing methods focus on single-step summarization.
    III. METHODOLOGY
    We evaluate on three benchmark datasets.
    """

    sections = split_into_sections(text)

    assert "introduction" in sections
    assert "literature_review" in sections
    assert "methodology" in sections
    assert "single-step summarization" in sections["literature_review"]


def test_extract_bibliography_handles_numbered_entries_and_continuations() -> None:
    sections = {
        "references": """
        [1] Smith, J. A long title for reference one
        continuing detail for the same source.
        [2] Lee, K. Reference two.
        """
    }

    refs = extract_bibliography(sections, "")

    assert len(refs) == 2
    assert refs[0].startswith("Smith, J.")
    assert "continuing detail" in refs[0]
    assert refs[1].startswith("Lee, K.")


def test_parse_paper_uses_text_extractor_output() -> None:
    fake_text = """
    Efficient Literature Mapping for Student Researchers
    Abstract
    This work studies practical mapping strategies.
    Introduction
    Mapping papers manually is repetitive.
    Results
    Structured extraction improves consistency.
    References
    [1] Miller, A. Mapping systems.
    """

    with patch("tools.paper_parser_tool.extract_text_from_pdf", return_value=fake_text):
        parsed = parse_paper("paper-1.pdf")

    assert parsed["title"].startswith("Efficient Literature Mapping")
    assert parsed["abstract"].startswith("This work studies")
    assert parsed["key_findings"]
    assert parsed["references"]


def test_split_into_sections_handles_inline_intro_marker_after_abstract_text() -> None:
    text = (
        "Abstract This paper proposes a practical workflow for local research support "
        "I. INTRODUCTION The toolchain improves traceability. "
        "II. METHODOLOGY We evaluate with two datasets."
    )

    sections = split_into_sections(text)

    assert "abstract" in sections
    assert "introduction" in sections
    assert "methodology" in sections
    assert "I. INTRODUCTION" not in sections["abstract"]


def test_parse_paper_trims_abstract_when_next_section_cue_is_embedded() -> None:
    noisy_text = """
    SpringForge: A Unified AI-Driven IntelliJ IDEA
    ABSTRACT
    Enterprise Spring Boot development is constrained by recurring bottlenecks across teams.
    Existing tools address issues in isolation.
    I. INTRODUCTION
    Spring Boot is widely adopted for cloud-native Java applications.
    """

    with patch("tools.paper_parser_tool.extract_text_from_pdf", return_value=noisy_text):
        parsed = parse_paper("paper-1.pdf")

    assert "introduction" in parsed["sections"]
    assert "Spring Boot is widely adopted" in parsed["sections"]["introduction"]
    assert "I. INTRODUCTION" not in parsed["abstract"]
    assert "Spring Boot is widely adopted" not in parsed["abstract"]


def test_literature_review_agent_sets_error_when_parser_fails() -> None:
    state = {
        "research_topic": "Automated synthesis",
        "paper_pdf_paths": ["paper-1.pdf"],
        "agent_logs": [],
        "errors": [],
    }

    with patch(
        "agents.literature_review_agent.parse_multiple_papers",
        side_effect=RuntimeError("parser exploded"),
    ):
        updated_state = literature_review_agent(state)

    assert updated_state.get("errors")
    assert any("Failed to parse papers" in err for err in updated_state["errors"])
    assert any(
        log.get("event_type") == "ERROR" and log.get("agent_name") == "literature_review_agent"
        for log in updated_state.get("agent_logs", [])
    )
