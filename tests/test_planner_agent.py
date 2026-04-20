"""
Tests for Member 1 - Research Planner Agent (Udula).

Unit tests  : mock pdfplumber and OllamaLLM - no Ollama required.
Integration : marked @pytest.mark.integration - requires Ollama running locally.

Run unit tests only  : pytest tests/ -m "not integration" -v
Run all tests        : pytest tests/ -v
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


TEST_ARTIFACTS_DIR = Path("outputs/test_artifacts")
MOCK_PDF_PATH = TEST_ARTIFACTS_DIR / "fake_proposal.pdf"
EMPTY_PDF_PATH = TEST_ARTIFACTS_DIR / "empty.pdf"

MOCK_PDF_TEXT = """Federated Learning for Healthcare Privacy Preservation

Keywords: federated learning, privacy, healthcare, deep learning, differential privacy

1. Introduction
This research investigates privacy-preserving machine learning in healthcare settings.

2. Objectives
1. To design a federated learning architecture for medical imaging.
2. To evaluate differential privacy mechanisms against baseline models.
3. To benchmark performance across at least five hospital nodes.
4. To publish findings as an open-source framework.

3. Scope
The scope covers hospital networks with at least 5 participating nodes using
de-identified patient data under HIPAA-compliant protocols.

4. Methodology
We employ federated averaging (FedAvg) with differential privacy noise injection.
Experiments are conducted on the CheXpert and MIMIC-CXR datasets.

5. References
[1] McMahan et al. (2017). Communication-efficient learning of deep networks.
[2] Dwork et al. (2014). The algorithmic foundations of differential privacy.
"""

MOCK_LLM_RESPONSE = json.dumps(
    {
        "implementation_plan": (
            "The research will proceed over 8 weeks. Weeks 1-2 focus on literature "
            "review and environment setup. Weeks 3-4 cover system design. Weeks 5-6 "
            "involve implementation and testing. Weeks 7-8 cover evaluation and reporting."
        ),
        "timeline": {
            "week_1_2": {
                "tasks": ["Literature review", "Install Ollama + dependencies"],
                "deliverable": "Literature summary",
            },
            "week_3_4": {
                "tasks": ["Design federated architecture", "Define privacy budget"],
                "deliverable": "Architecture document",
            },
            "week_5_6": {
                "tasks": ["Implement FedAvg", "Run initial experiments"],
                "deliverable": "Working prototype",
            },
            "week_7_8": {
                "tasks": ["Evaluate results", "Write final report"],
                "deliverable": "Final paper draft",
            },
        },
        "suggested_resources": [
            "IEEE Xplore",
            "ACM Digital Library",
            "arXiv.org",
            "PapersWithCode",
            "Semantic Scholar",
            "TensorFlow Federated",
        ],
    }
)


@pytest.fixture(scope="module", autouse=True)
def prepare_test_artifacts():
    """Create repo-local placeholder PDFs used by patched tool tests."""
    TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MOCK_PDF_PATH.touch(exist_ok=True)
    EMPTY_PDF_PATH.touch(exist_ok=True)


@pytest.fixture
def mock_pdf():
    """Patch pdfplumber.open to return MOCK_PDF_TEXT from a fake PDF."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = MOCK_PDF_TEXT

    mock_pdf_obj = MagicMock()
    mock_pdf_obj.__enter__ = MagicMock(return_value=mock_pdf_obj)
    mock_pdf_obj.__exit__ = MagicMock(return_value=False)
    mock_pdf_obj.pages = [mock_page]

    with patch("pdfplumber.open", return_value=mock_pdf_obj):
        yield MOCK_PDF_PATH


@pytest.fixture
def initial_state() -> Dict[str, Any]:
    """Minimal ResearchMindState for agent tests."""
    return {
        "research_topic": "Federated Learning for Healthcare Privacy Preservation",
        "proposal_pdf_path": str(MOCK_PDF_PATH),
        "paper_pdf_paths": [],
        "own_paper_path": "",
        "model_name": "gemma4:e2b",
        "proposal_extracted": {},
        "implementation_plan": "",
        "timeline": {},
        "suggested_resources": [],
        "paper_summaries": [],
        "citation_map": {},
        "core_themes": [],
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


class TestProposalAnalyzerTool:
    def test_returns_all_required_keys(self, mock_pdf):
        from tools.proposal_analyzer_tool import proposal_analyzer_tool

        result = proposal_analyzer_tool(str(mock_pdf))
        required = {"title", "objectives", "scope", "methodology", "keywords", "raw_text"}
        missing = required - result.keys()
        assert not missing, f"Missing keys in output: {missing}"

    def test_objectives_is_list(self, mock_pdf):
        from tools.proposal_analyzer_tool import proposal_analyzer_tool

        result = proposal_analyzer_tool(str(mock_pdf))
        assert isinstance(result["objectives"], list)

    def test_keywords_is_list(self, mock_pdf):
        from tools.proposal_analyzer_tool import proposal_analyzer_tool

        result = proposal_analyzer_tool(str(mock_pdf))
        assert isinstance(result["keywords"], list)

    def test_keywords_extracted_correctly(self, mock_pdf):
        from tools.proposal_analyzer_tool import proposal_analyzer_tool

        result = proposal_analyzer_tool(str(mock_pdf))
        assert len(result["keywords"]) >= 3, (
            f"Expected >= 3 keywords, got {result['keywords']}"
        )

    def test_raw_text_is_non_empty(self, mock_pdf):
        from tools.proposal_analyzer_tool import proposal_analyzer_tool

        result = proposal_analyzer_tool(str(mock_pdf))
        assert result["raw_text"].strip(), "raw_text should not be empty"

    def test_raises_file_not_found(self):
        from tools.proposal_analyzer_tool import proposal_analyzer_tool

        with pytest.raises(FileNotFoundError):
            proposal_analyzer_tool("non_existent_path.pdf")

    def test_raises_value_error_on_empty_pdf(self):
        from tools.proposal_analyzer_tool import proposal_analyzer_tool

        empty_page = MagicMock()
        empty_page.extract_text.return_value = ""
        mock_pdf_obj = MagicMock()
        mock_pdf_obj.__enter__ = MagicMock(return_value=mock_pdf_obj)
        mock_pdf_obj.__exit__ = MagicMock(return_value=False)
        mock_pdf_obj.pages = [empty_page]

        with patch("pdfplumber.open", return_value=mock_pdf_obj):
            with pytest.raises(ValueError, match="no extractable text"):
                proposal_analyzer_tool(str(EMPTY_PDF_PATH))


class TestResearchPlannerAgent:
    def _run_agent_with_mocks(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Helper: run research_planner_agent with mocked tool and LLM."""
        mock_extracted = {
            "title": "Federated Learning for Healthcare Privacy Preservation",
            "objectives": [
                "Design a federated learning architecture for medical imaging.",
                "Evaluate differential privacy mechanisms.",
                "Benchmark performance across hospital nodes.",
                "Publish findings as open-source.",
            ],
            "scope": "Hospital networks with 5+ participating nodes.",
            "methodology": "FedAvg with differential privacy noise injection.",
            "keywords": ["federated learning", "privacy", "healthcare"],
            "raw_text": MOCK_PDF_TEXT,
        }

        with patch(
            "agents.research_planner_agent.proposal_analyzer_tool",
            return_value=mock_extracted,
        ):
            with patch("agents.research_planner_agent.OllamaLLM") as mock_llm:
                mock_llm.return_value.invoke.return_value = MOCK_LLM_RESPONSE
                from agents.research_planner_agent import research_planner_agent

                return research_planner_agent(initial_state)

    def test_timeline_has_at_least_four_milestones(self, initial_state):
        result = self._run_agent_with_mocks(initial_state)
        assert len(result["timeline"]) >= 4, (
            f"Expected >= 4 timeline milestones, got {len(result['timeline'])}: "
            f"{list(result['timeline'].keys())}"
        )

    def test_suggested_resources_is_nonempty(self, initial_state):
        result = self._run_agent_with_mocks(initial_state)
        assert isinstance(result["suggested_resources"], list)
        assert len(result["suggested_resources"]) > 0, "suggested_resources must not be empty"

    def test_implementation_plan_is_nonempty_string(self, initial_state):
        result = self._run_agent_with_mocks(initial_state)
        assert isinstance(result["implementation_plan"], str)
        assert len(result["implementation_plan"]) > 0, "implementation_plan must not be empty"

    def test_proposal_extracted_populated(self, initial_state):
        result = self._run_agent_with_mocks(initial_state)
        assert result["proposal_extracted"], "proposal_extracted should not be empty after agent run"

    def test_selected_model_is_used_for_ollama_call(self, initial_state):
        mock_extracted = {
            "title": "Test",
            "objectives": [],
            "scope": "",
            "methodology": "",
            "keywords": [],
            "raw_text": MOCK_PDF_TEXT,
        }
        with patch(
            "agents.research_planner_agent.proposal_analyzer_tool",
            return_value=mock_extracted,
        ):
            with patch("agents.research_planner_agent.OllamaLLM") as mock_llm:
                mock_llm.return_value.invoke.return_value = MOCK_LLM_RESPONSE
                from agents.research_planner_agent import research_planner_agent

                research_planner_agent(initial_state)

        mock_llm.assert_called_once_with(model="gemma4:e2b", temperature=0.3)

    def test_agent_logs_contain_start_and_output(self, initial_state):
        result = self._run_agent_with_mocks(initial_state)
        event_types = [log["event_type"] for log in result["agent_logs"]]
        assert "START" in event_types, "Agent log missing START event"
        assert "OUTPUT" in event_types, "Agent log missing OUTPUT event"

    def test_no_errors_on_valid_input(self, initial_state):
        result = self._run_agent_with_mocks(initial_state)
        assert result["errors"] == [], f"Unexpected errors: {result['errors']}"

    def test_error_appended_on_missing_pdf(self, initial_state):
        initial_state["proposal_pdf_path"] = "missing_file.pdf"
        from agents.research_planner_agent import research_planner_agent

        result = research_planner_agent(initial_state)
        assert len(result["errors"]) > 0, "Expected an error for missing PDF"
        assert "research_planner_agent" in result["errors"][0]

    def test_fallback_used_when_llm_returns_garbage(self, initial_state):
        mock_extracted = {
            "title": "Test",
            "objectives": [],
            "scope": "",
            "methodology": "",
            "keywords": [],
            "raw_text": "test content",
        }
        with patch(
            "agents.research_planner_agent.proposal_analyzer_tool",
            return_value=mock_extracted,
        ):
            with patch("agents.research_planner_agent.OllamaLLM") as mock_llm:
                mock_llm.return_value.invoke.return_value = "THIS IS NOT JSON AT ALL !!!"
                from agents.research_planner_agent import research_planner_agent

                result = research_planner_agent(initial_state)

        assert len(result["timeline"]) >= 4
        assert len(result["suggested_resources"]) > 0


@pytest.mark.integration
class TestLLMJudge:
    def test_implementation_plan_is_coherent_and_relevant(self, initial_state, mock_pdf):
        """
        Integration test: runs the full agent (real Ollama required), then asks
        the configured Ollama model to judge whether the plan is coherent and topic-relevant.

        Skip in CI with: pytest -m "not integration"
        """
        from agents.research_planner_agent import research_planner_agent
        from langchain_ollama import OllamaLLM

        initial_state["proposal_pdf_path"] = str(mock_pdf)

        result = research_planner_agent(initial_state)
        assert result["errors"] == [], f"Agent errors: {result['errors']}"

        plan = result["implementation_plan"]
        topic = initial_state["research_topic"]

        judge_llm = OllamaLLM(model=initial_state["model_name"], temperature=0.0)
        judge_prompt = (
            f'Research topic: "{topic}"\n'
            f'Implementation plan: "{plan[:1000]}"\n\n'
            "Evaluate the plan. Respond ONLY with valid JSON:\n"
            '{"coherent": true_or_false, "topic_relevant": true_or_false, '
            '"reason": "one sentence"}'
        )
        raw = judge_llm.invoke(judge_prompt)

        try:
            import re

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            judgment = json.loads(match.group() if match else raw)
        except Exception:
            pytest.fail(f"Judge LLM did not return valid JSON. Raw: {raw}")

        assert judgment.get("coherent") is True, (
            f"LLM judge: plan not coherent - {judgment.get('reason')}"
        )
        assert judgment.get("topic_relevant") is True, (
            f"LLM judge: plan not topic-relevant - {judgment.get('reason')}"
        )
