"""
Tests for Member 3 — Research Gap & Hypothesis Agent (Madhini).

Unit tests: gap_analyzer_tool is fully deterministic — no Ollama required.
Integration: gap_hypothesis_agent is also deterministic (no LLM), so all
tests here run without Ollama.

Run: pytest tests/test_gap_agent.py -v
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

PAPER_A = {
    "title": "Paper A",
    "abstract": "We study accuracy and scalability of neural networks.",
    "key_findings": [
        "The proposed method improves accuracy significantly across all benchmarks.",
        "Our approach outperforms prior baselines on efficiency metrics.",
    ],
    "sections": {},
    "references": [],
}

PAPER_B = {
    "title": "Paper B",
    "abstract": "A comparative study of deep learning classification methods.",
    "key_findings": [
        "The model degrades accuracy under adversarial noise conditions.",
        "No improvement was observed for efficiency in low-resource settings.",
    ],
    "sections": {},
    "references": [],
}

PAPER_C = {
    "title": "Paper C",
    "abstract": "Federated learning for privacy-preserving healthcare analytics.",
    "key_findings": [
        "Privacy and scalability remain open questions in federated contexts.",
        "Future work should investigate generalization to cross-domain settings.",
        "Limited real-world deployment studies exist in this area.",
    ],
    "sections": {},
    "references": [],
}

MOCK_CORE_THEMES = ["deep learning", "privacy", "scalability"]
MOCK_CITATION_MAP = {"deep learning": 3, "privacy": 1}
MOCK_RESEARCH_TOPIC = "AI-Driven Healthcare Systems"


def _make_state(**overrides) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "research_topic": MOCK_RESEARCH_TOPIC,
        "proposal_pdf_path": "",
        "paper_pdf_paths": [],
        "own_paper_path": "",
        "model_name": "gemma4:e2b",
        "proposal_extracted": {},
        "implementation_plan": "",
        "timeline": {},
        "suggested_resources": [],
        "paper_summaries": [PAPER_A, PAPER_B, PAPER_C],
        "citation_map": MOCK_CITATION_MAP,
        "core_themes": MOCK_CORE_THEMES,
        "section_explanations": {},
        "literature_review_report": {},
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
        "final_report_path": "",
        "agent_logs": [],
        "errors": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests for gap_analyzer_tool (deterministic, no mocking needed)
# ---------------------------------------------------------------------------

class TestGapAnalyzerTool:
    def test_returns_all_required_keys(self):
        from tools.gap_analyzer_tool import gap_analyzer_tool

        result = gap_analyzer_tool(
            paper_summaries=[PAPER_A],
            citation_map=MOCK_CITATION_MAP,
            core_themes=MOCK_CORE_THEMES,
            research_topic=MOCK_RESEARCH_TOPIC,
        )
        required = {
            "identified_gaps",
            "gap_frequency_scores",
            "hypotheses",
            "positioning_statement",
            "contradiction_pairs",
        }
        missing = required - result.keys()
        assert not missing, f"Missing keys in output: {missing}"

    def test_contradiction_detected_when_papers_disagree(self):
        """Paper A is positive about accuracy; Paper B is negative — must detect contradiction."""
        from tools.gap_analyzer_tool import gap_analyzer_tool

        result = gap_analyzer_tool(
            paper_summaries=[PAPER_A, PAPER_B],
            citation_map=MOCK_CITATION_MAP,
            core_themes=MOCK_CORE_THEMES,
            research_topic=MOCK_RESEARCH_TOPIC,
        )
        assert len(result["contradiction_pairs"]) >= 1, (
            f"Expected at least one contradiction pair, got: {result['contradiction_pairs']}"
        )
        aspects = {p["aspect"] for p in result["contradiction_pairs"]}
        assert aspects & {"accuracy", "efficiency"}, (
            f"Expected contradiction on accuracy or efficiency, got aspects: {aspects}"
        )

    def test_positioning_statement_is_nonempty_string(self):
        from tools.gap_analyzer_tool import gap_analyzer_tool

        result = gap_analyzer_tool(
            paper_summaries=[PAPER_A, PAPER_C],
            citation_map=MOCK_CITATION_MAP,
            core_themes=MOCK_CORE_THEMES,
            research_topic=MOCK_RESEARCH_TOPIC,
        )
        assert isinstance(result["positioning_statement"], str)
        assert result["positioning_statement"].strip(), "positioning_statement must not be empty"

    def test_at_least_two_hypotheses_generated(self):
        from tools.gap_analyzer_tool import gap_analyzer_tool

        result = gap_analyzer_tool(
            paper_summaries=[PAPER_A, PAPER_B, PAPER_C],
            citation_map=MOCK_CITATION_MAP,
            core_themes=MOCK_CORE_THEMES,
            research_topic=MOCK_RESEARCH_TOPIC,
        )
        assert len(result["hypotheses"]) >= 2, (
            f"Expected >= 2 hypotheses, got: {result['hypotheses']}"
        )

    def test_gap_frequency_scores_nonempty(self):
        from tools.gap_analyzer_tool import gap_analyzer_tool

        result = gap_analyzer_tool(
            paper_summaries=[PAPER_C],
            citation_map={},
            core_themes=["generalization", "privacy"],
            research_topic=MOCK_RESEARCH_TOPIC,
        )
        assert isinstance(result["gap_frequency_scores"], dict)
        assert len(result["gap_frequency_scores"]) > 0, "gap_frequency_scores must not be empty"

    def test_empty_input_returns_fallback_gaps(self):
        """Tool should not crash and must return fallback gaps when given empty input."""
        from tools.gap_analyzer_tool import gap_analyzer_tool

        result = gap_analyzer_tool(
            paper_summaries=[],
            citation_map={},
            core_themes=[],
            research_topic=MOCK_RESEARCH_TOPIC,
        )
        assert len(result["identified_gaps"]) >= 1, (
            "Should produce fallback gaps even for empty input"
        )
        assert len(result["hypotheses"]) >= 1, (
            "Should produce fallback hypotheses even for empty input"
        )


# ---------------------------------------------------------------------------
# Tests for gap_hypothesis_agent (patches the tool so tests remain fast)
# ---------------------------------------------------------------------------

MOCK_TOOL_RESULT = {
    "identified_gaps": [
        "In AI-Driven Healthcare Systems, current literature shows limited cross-domain generalization.",
        "In AI-Driven Healthcare Systems, current literature shows incomplete privacy and security analysis.",
    ],
    "gap_frequency_scores": {"generalization": 1.0, "privacy_security": 0.8},
    "hypotheses": [
        "H1: Addressing limited cross-domain generalization will improve outcomes related to deep learning.",
        "H2: Addressing privacy gaps will improve security outcomes related to privacy.",
    ],
    "positioning_statement": (
        "This study positions itself at the intersection of AI-Driven Healthcare Systems "
        "and deep learning, specifically targeting limited cross-domain generalization "
        "through a measurable and reproducible methodology."
    ),
    "contradiction_pairs": [
        {
            "aspect": "accuracy",
            "paper_a": "Paper A",
            "claim_a": "improves accuracy significantly",
            "paper_b": "Paper B",
            "claim_b": "degrades accuracy under adversarial noise",
        }
    ],
}


class TestGapHypothesisAgent:
    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        with patch(
            "agents.gap_hypothesis_agent.gap_analyzer_tool",
            return_value=MOCK_TOOL_RESULT,
        ):
            from agents.gap_hypothesis_agent import gap_hypothesis_agent
            return gap_hypothesis_agent(state)

    def test_all_five_output_fields_populated(self):
        result = self._run(_make_state())
        assert result["identified_gaps"], "identified_gaps must not be empty"
        assert result["gap_frequency_scores"], "gap_frequency_scores must not be empty"
        assert result["hypotheses"], "hypotheses must not be empty"
        assert result["positioning_statement"], "positioning_statement must not be empty"
        assert isinstance(result["contradiction_pairs"], list)

    def test_at_least_two_hypotheses_in_state(self):
        result = self._run(_make_state())
        assert len(result["hypotheses"]) >= 2, (
            f"Expected >= 2 hypotheses in state, got: {result['hypotheses']}"
        )

    def test_contradiction_pairs_written_to_state(self):
        result = self._run(_make_state())
        assert len(result["contradiction_pairs"]) >= 1, (
            "Expected at least one contradiction pair in state"
        )

    def test_agent_logs_contain_start_and_output(self):
        result = self._run(_make_state())
        event_types = [log["event_type"] for log in result["agent_logs"]]
        assert "START" in event_types, "Agent log missing START event"
        assert "OUTPUT" in event_types, "Agent log missing OUTPUT event"

    def test_no_errors_on_valid_input(self):
        result = self._run(_make_state())
        assert result["errors"] == [], f"Unexpected errors: {result['errors']}"

    def test_prior_state_fields_preserved(self):
        """Agent must not overwrite fields written by earlier agents."""
        state = _make_state(
            implementation_plan="Week 1: Literature review",
            paper_summaries=[PAPER_A],
        )
        result = self._run(state)
        assert result["implementation_plan"] == "Week 1: Literature review", (
            "Agent 3 must not overwrite Agent 1's implementation_plan"
        )
        assert result["paper_summaries"] == [PAPER_A], (
            "Agent 3 must not overwrite Agent 2's paper_summaries"
        )
