from __future__ import annotations

from typing import Any, Dict


def _initial_state() -> Dict[str, Any]:
    return {
        "research_topic": "Federated Learning for Healthcare",
        "proposal_pdf_path": "",
        "paper_pdf_paths": [],
        "own_paper_path": "",
        "model_name": "gemma4:e2b",
        "proposal_extracted": {},
        "implementation_plan": "",
        "timeline": {},
        "suggested_resources": [],
        "paper_summaries": [
            {
                "title": "Paper A",
                "summary": (
                    "Our method improves accuracy but has limited generalization under domain shift. "
                    "Future work should evaluate real-world deployment."
                ),
            },
            {
                "title": "Paper B",
                "summary": (
                    "In cross-hospital testing there is no improvement in accuracy and some performance drop. "
                    "The method remains robust to noise."
                ),
            },
        ],
        "citation_map": {
            "generalization": 1,
            "privacy": 4,
            "real_world_validation": {"citations": ["c1"]},
        },
        "core_themes": ["privacy-preserving training", "cross-hospital evaluation"],
        "section_explanations": {},
        "identified_gaps": [],
        "gap_frequency_scores": {},
        "hypotheses": [],
        "positioning_statement": "",
        "contradiction_pairs": [],
        "formatting_issues": [],
        "missing_sections": [],
        "plagiarism_score": 0.0,
        "integrity_score": 0.0,
        "audit_feedback": "",
        "final_report_path": "outputs/report.json",
        "agent_logs": [],
        "errors": [],
    }


def test_gap_analyzer_tool_returns_all_required_keys():
    from tools.gap_analyzer_tool import gap_analyzer_tool

    state = _initial_state()
    result = gap_analyzer_tool(
        paper_summaries=state["paper_summaries"],
        citation_map=state["citation_map"],
        core_themes=state["core_themes"],
        research_topic=state["research_topic"],
    )

    required = {
        "identified_gaps",
        "gap_frequency_scores",
        "hypotheses",
        "positioning_statement",
        "contradiction_pairs",
    }
    assert required.issubset(result.keys())


def test_gap_hypothesis_agent_populates_state_fields():
    from agents.gap_hypothesis_agent import gap_hypothesis_agent

    result = gap_hypothesis_agent(_initial_state())

    assert len(result["identified_gaps"]) > 0
    assert len(result["gap_frequency_scores"]) > 0
    assert len(result["hypotheses"]) > 0
    assert isinstance(result["positioning_statement"], str)
    assert result["positioning_statement"].strip() != ""
    assert isinstance(result["contradiction_pairs"], list)


def test_gap_hypothesis_agent_logs_start_tool_output():
    from agents.gap_hypothesis_agent import gap_hypothesis_agent

    result = gap_hypothesis_agent(_initial_state())
    event_types = [log["event_type"] for log in result["agent_logs"]]

    assert "START" in event_types
    assert "TOOL_CALL" in event_types
    assert "OUTPUT" in event_types


def test_gap_hypothesis_agent_detects_contradiction_pair():
    from agents.gap_hypothesis_agent import gap_hypothesis_agent

    result = gap_hypothesis_agent(_initial_state())

    assert len(result["contradiction_pairs"]) >= 1
    assert "aspect" in result["contradiction_pairs"][0]


def test_gap_hypothesis_agent_handles_empty_inputs_without_error():
    from agents.gap_hypothesis_agent import gap_hypothesis_agent

    state = _initial_state()
    state["paper_summaries"] = []
    state["citation_map"] = {}
    state["core_themes"] = []

    result = gap_hypothesis_agent(state)

    assert result["errors"] == []
    assert len(result["identified_gaps"]) > 0
    assert len(result["hypotheses"]) > 0
