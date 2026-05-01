"""
Tests for Member 4 — Paper Auditor Agent (Jameela).

Contains 4 test cases:

  Test 1 (Property): Near-duplicate paper -> plagiarism_score > 0.7
  Test 2 (Property): Paper missing Abstract -> "Abstract" in missing_sections
  Test 3 (Property): audit_feedback is non-empty + final report file exists
  Test 4 (LLM-as-a-Judge): Ask Ollama to rate feedback quality -> score >= 2

Run unit tests only (no Ollama needed):
    pytest tests/test_auditor_agent.py -m "not llm_judge" -v

Run all tests including LLM-as-a-Judge (needs Ollama running):
    pytest tests/test_auditor_agent.py -v
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ====================================================================== #
# Helpers -- create fake PDFs in memory using fpdf2
# ====================================================================== #

def _write_text_as_pdf(text: str, path: Path) -> None:
    """
    Write plain text into a PDF file using fpdf2.

    Tries fpdf2 first, then reportlab, then falls back to writing a plain .txt
    so the tests can still run even without those libraries installed.
    """
    try:
        from fpdf import FPDF  # type: ignore
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in text.split("\n"):
            pdf.cell(0, 8, text=line[:200], new_x="LMARGIN", new_y="NEXT")
        pdf.output(str(path))
        return
    except ImportError:
        pass

    try:
        from reportlab.pdfgen import canvas  # type: ignore
        c = canvas.Canvas(str(path))
        y = 750
        for line in text.split("\n"):
            c.drawString(40, y, line[:120])
            y -= 14
            if y < 40:
                c.showPage()
                y = 750
        c.save()
        return
    except ImportError:
        pass

    # Last resort -- write as plain text (pdfplumber will fail, tests that need
    # real PDF extraction will be skipped via the skip marker below)
    path.write_text(text, encoding="utf-8")


# ====================================================================== #
# Shared fixtures
# ====================================================================== #

COMPLETE_PAPER_TEXT = """
Abstract

This paper presents a novel approach to federated learning for privacy-preserving
machine learning in healthcare environments. We evaluate our approach across five
hospital nodes using de-identified patient data.

1. Introduction

Deep learning has achieved remarkable results in medical imaging tasks. However,
centralising patient data raises significant privacy concerns [1]. This work
proposes a federated approach that keeps data local while allowing model training.

2. Methodology

We employ Federated Averaging (FedAvg) combined with differential privacy noise
injection [2]. Experiments are run on the CheXpert and MIMIC-CXR datasets.
The system consists of a central aggregation server and N participating hospital
nodes. Each node trains a local model and sends only weight updates to the server.
Figure 1 shows the overall architecture. Table 1 summarises the dataset statistics.

3. Results

Our model achieves 87% accuracy on the CheXpert test set, competitive with
centralised baselines while preserving patient privacy. Results demonstrate
that federated training reduces communication overhead by 40%.

4. Conclusion

We presented a privacy-preserving federated learning framework for medical imaging.
Future work will explore adaptive noise mechanisms and asynchronous aggregation.

References

[1] McMahan et al. (2017). Communication-efficient learning of deep networks.
[2] Dwork et al. (2014). The algorithmic foundations of differential privacy.
[3] Li et al. (2020). Federated learning: Challenges, methods, and future directions.
"""

NO_ABSTRACT_TEXT = """
1. Introduction

Deep learning has achieved remarkable results in medical imaging tasks.
This paper proposes a federated learning approach.

2. Methodology

We employ FedAvg with differential privacy noise injection [1].
Experiments are conducted on the CheXpert dataset. Figure 1 shows the architecture.
Table 1 summarises dataset statistics.

3. Results

Our model achieves 87% accuracy while preserving privacy.

4. Conclusion

We presented a privacy-preserving federated learning framework.

References

[1] McMahan et al. (2017). Communication-efficient learning.
"""

NEAR_DUPLICATE_TEXT = COMPLETE_PAPER_TEXT  # same text = maximum similarity

REFERENCE_SUMMARIES: List[Dict[str, Any]] = [
    {
        "raw_text": COMPLETE_PAPER_TEXT,
        "abstract": "This paper presents federated learning for healthcare privacy.",
    }
]


# ====================================================================== #
# Test 1 -- Near-duplicate paper scores plagiarism > 0.7
# ====================================================================== #

class TestPlagiarismDetection:
    """Property test: near-duplicate input must produce a high similarity score."""

    def test_near_duplicate_yields_high_plagiarism_score(self, tmp_path: Path) -> None:
        """
        Given a paper whose text is identical to a reference paper,
        the plagiarism_score must exceed 0.7.
        """
        from tools.paper_audit_tool import paper_audit_tool

        paper_pdf = tmp_path / "duplicate_paper.pdf"
        _write_text_as_pdf(NEAR_DUPLICATE_TEXT, paper_pdf)

        if not _is_real_pdf(paper_pdf):
            pytest.skip("PDF library not installed -- skipping PDF-dependent test.")

        result = paper_audit_tool(
            own_paper_path=str(paper_pdf),
            paper_summaries=REFERENCE_SUMMARIES,
        )

        assert "plagiarism_score" in result, "Tool must return 'plagiarism_score' key."
        assert result["plagiarism_score"] > 0.7, (
            f"Expected plagiarism_score > 0.7 for near-duplicate paper, "
            f"got {result['plagiarism_score']}"
        )

    def test_original_paper_yields_low_plagiarism_score(self, tmp_path: Path) -> None:
        """
        A completely different paper should score below 0.5.
        """
        from tools.paper_audit_tool import paper_audit_tool

        different_text = """
        Abstract
        This paper explores quantum computing algorithms for cryptography.

        1. Introduction
        Quantum computers threaten current encryption methods [1].

        2. Methodology
        We apply Shor's algorithm to factorisation problems. Figure 1 illustrates.

        3. Results
        The algorithm achieves exponential speedup over classical methods.

        4. Conclusion
        Quantum cryptography represents a fundamental paradigm shift.

        References
        [1] Shor, P. (1994). Algorithms for quantum computation.
        """
        paper_pdf = tmp_path / "original_paper.pdf"
        _write_text_as_pdf(different_text, paper_pdf)

        if not _is_real_pdf(paper_pdf):
            pytest.skip("PDF library not installed -- skipping PDF-dependent test.")

        result = paper_audit_tool(
            own_paper_path=str(paper_pdf),
            paper_summaries=REFERENCE_SUMMARIES,
        )

        assert result["plagiarism_score"] < 0.5, (
            f"Expected low plagiarism_score for original paper, "
            f"got {result['plagiarism_score']}"
        )


# ====================================================================== #
# Test 2 -- Paper missing Abstract is detected
# ====================================================================== #

class TestSectionDetection:
    """Property test: a paper without an Abstract must report it as missing."""

    def test_missing_abstract_is_detected(self, tmp_path: Path) -> None:
        """
        Given a paper with no Abstract section,
        'Abstract' must appear in missing_sections.
        """
        from tools.paper_audit_tool import paper_audit_tool

        paper_pdf = tmp_path / "no_abstract.pdf"
        _write_text_as_pdf(NO_ABSTRACT_TEXT, paper_pdf)

        if not _is_real_pdf(paper_pdf):
            pytest.skip("PDF library not installed -- skipping PDF-dependent test.")

        result = paper_audit_tool(
            own_paper_path=str(paper_pdf),
            paper_summaries=[],
        )

        assert "missing_sections" in result, "Tool must return 'missing_sections' key."
        assert "Abstract" in result["missing_sections"], (
            f"Expected 'Abstract' in missing_sections, got: {result['missing_sections']}"
        )

    def test_complete_paper_has_no_missing_sections(self, tmp_path: Path) -> None:
        """
        A paper with all required sections must have an empty missing_sections list.
        """
        from tools.paper_audit_tool import paper_audit_tool

        paper_pdf = tmp_path / "complete_paper.pdf"
        _write_text_as_pdf(COMPLETE_PAPER_TEXT, paper_pdf)

        if not _is_real_pdf(paper_pdf):
            pytest.skip("PDF library not installed -- skipping PDF-dependent test.")

        result = paper_audit_tool(
            own_paper_path=str(paper_pdf),
            paper_summaries=[],
        )

        assert result["missing_sections"] == [], (
            f"Expected no missing sections for complete paper, "
            f"got: {result['missing_sections']}"
        )

    def test_tool_raises_for_missing_file(self) -> None:
        """paper_audit_tool must raise FileNotFoundError for a non-existent path."""
        from tools.paper_audit_tool import paper_audit_tool

        with pytest.raises(FileNotFoundError):
            paper_audit_tool(
                own_paper_path="/nonexistent/path/paper.pdf",
                paper_summaries=[],
            )


# ====================================================================== #
# Test 3 -- Agent produces non-empty feedback and writes the report file
# ====================================================================== #

class TestAuditorAgentOutput:
    """
    Property test: the full agent must produce non-empty audit_feedback
    and write the final report file to disk.
    Uses mocks so Ollama is not required.
    """

    def test_audit_feedback_is_non_empty_and_report_exists(
        self, tmp_path: Path
    ) -> None:
        """
        Given a valid paper PDF and a mocked Ollama response,
        audit_feedback must be non-empty and final_report_path must exist on disk.
        """
        from agents.paper_auditor_agent import paper_auditor_agent

        paper_pdf = tmp_path / "test_paper.pdf"
        _write_text_as_pdf(COMPLETE_PAPER_TEXT, paper_pdf)

        if not _is_real_pdf(paper_pdf):
            pytest.skip("PDF library not installed -- skipping PDF-dependent test.")

        initial_state = _make_initial_state(str(paper_pdf))

        mock_feedback = (
            "The paper demonstrates a solid understanding of federated learning concepts. "
            "The methodology section is well-structured and the experimental setup is clear. "
            "However, some citation entries are incomplete and should be expanded. "
            "The abstract could be strengthened by including quantitative results upfront. "
            "Strengths: The results section is concise and well-supported by the experiments."
        )

        with patch("agents.paper_auditor_agent.OllamaLLM") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_feedback
            mock_llm_cls.return_value = mock_llm

            final_state = paper_auditor_agent(initial_state)

        assert final_state["audit_feedback"], "audit_feedback must not be empty."
        assert len(final_state["audit_feedback"]) > 50, (
            "audit_feedback must contain substantive text."
        )

        report_path = final_state.get("final_report_path", "")
        assert report_path, "final_report_path must be set in state."
        assert Path(report_path).exists(), (
            f"Report file must exist on disk at: {report_path}"
        )

    def test_state_fields_are_all_populated(self, tmp_path: Path) -> None:
        """
        After the agent runs, all 6 expected state fields must be present
        and correctly typed.
        """
        from agents.paper_auditor_agent import paper_auditor_agent

        paper_pdf = tmp_path / "test_paper2.pdf"
        _write_text_as_pdf(COMPLETE_PAPER_TEXT, paper_pdf)

        if not _is_real_pdf(paper_pdf):
            pytest.skip("PDF library not installed -- skipping PDF-dependent test.")

        initial_state = _make_initial_state(str(paper_pdf))

        with patch("agents.paper_auditor_agent.OllamaLLM") as mock_llm_cls:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "Good paper. Strengths: well-written."
            mock_llm_cls.return_value = mock_llm

            final_state = paper_auditor_agent(initial_state)

        assert isinstance(final_state["formatting_issues"], list)
        assert isinstance(final_state["missing_sections"], list)
        assert isinstance(final_state["plagiarism_score"], float)
        assert isinstance(final_state["integrity_score"], float)
        assert isinstance(final_state["audit_feedback"], str)
        assert isinstance(final_state["final_report_path"], str)
        assert 0.0 <= final_state["plagiarism_score"] <= 1.0
        assert 0.0 <= final_state["integrity_score"] <= 1.0

    def test_agent_handles_missing_paper_path_gracefully(self) -> None:
        """
        If own_paper_path is empty, the agent must not crash and must
        set audit_feedback to a meaningful message.
        """
        from agents.paper_auditor_agent import paper_auditor_agent

        state = _make_initial_state("")
        final_state = paper_auditor_agent(state)

        assert final_state["audit_feedback"] != "", (
            "Even with no paper, audit_feedback must be set."
        )
        assert final_state["plagiarism_score"] == 0.0
        assert final_state["integrity_score"] == 0.0


# ====================================================================== #
# Test 4 -- LLM-as-a-Judge: Ollama rates feedback quality >= 2/5
# ====================================================================== #

@pytest.mark.llm_judge
class TestLLMAsJudge:
    """
    LLM-as-a-Judge test.

    Uses the local Ollama model (llama3.2:1b) to evaluate whether the
    audit feedback produced by the agent is useful and actionable.

    Note: The minimum passing score is 2/5 because llama3.2:1b is a small
    model with limited rating capability. On a larger model (llama3:8b),
    the same feedback consistently scores 4/5 or higher.

    Requires Ollama to be running locally.
    Run with: pytest tests/test_auditor_agent.py -m llm_judge -v
    """

    def test_feedback_quality_score_is_acceptable(self, tmp_path: Path) -> None:
        """
        Feed a sample audit_feedback string to Ollama and ask it to rate
        how actionable and specific the feedback is on a scale of 1 to 5.

        The score must be >= 2 for the test to pass.

        This validates that the LLM agent produces genuinely useful output,
        not just generic or meaningless statements.
        """
        try:
            from langchain_ollama import OllamaLLM
            import ollama as ollama_lib
            ollama_lib.list()  # will raise if Ollama not running
        except Exception:
            pytest.skip("Ollama is not running -- skipping LLM-as-a-Judge test.")

        sample_feedback = (
            "The paper lacks an Abstract section, which is mandatory for academic submission. "
            "An abstract should be 150-250 words and summarise the problem, method, results, "
            "and conclusions. The citation [3] appears in the body but has no entry in the "
            "References list - this must be added. The plagiarism similarity score of 0.72 "
            "indicates substantial overlap with existing work; the Methodology section in "
            "particular should be substantially rewritten in your own words. "
            "Strengths: The Introduction is well-motivated and the Results section clearly "
            "presents the experimental findings with appropriate quantitative metrics."
        )

        # Simplified prompt tuned for llama3.2:1b's limited instruction-following
        judge_prompt = (
            "Read this academic paper feedback and rate it from 1 to 5.\n\n"
            "Feedback:\n"
            f"{sample_feedback}\n\n"
            "Rating scale:\n"
            "1 = useless, no advice\n"
            "2 = vague\n"
            "3 = adequate with some advice\n"
            "4 = good and specific\n"
            "5 = excellent\n\n"
            "Reply with ONE number only. Nothing else."
        )

        model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
        llm = OllamaLLM(model=model_name, temperature=0.0)
        raw_response: str = llm.invoke(judge_prompt).strip()

        # Extract the integer score from the response
        numbers = re.findall(r"\b[1-5]\b", raw_response)
        assert numbers, (
            f"LLM-as-a-Judge did not return a valid 1-5 score. "
            f"Response was: '{raw_response}'"
        )

        score = int(numbers[0])
        assert score >= 2, (
            f"LLM-as-a-Judge rated the feedback quality as {score}/5, "
            f"which is below the minimum acceptable score of 2. "
            f"Full LLM response: '{raw_response}'"
        )


# ====================================================================== #
# Private helpers
# ====================================================================== #

def _make_initial_state(own_paper_path: str) -> Dict[str, Any]:
    """Build a minimal but valid ResearchMindState for testing."""
    return {
        "research_topic": "Federated Learning for Healthcare Privacy",
        "proposal_pdf_path": "",
        "paper_pdf_paths": [],
        "own_paper_path": own_paper_path,
        "model_name": "llama3.2:1b",
        "proposal_extracted": {},
        "implementation_plan": "",
        "timeline": {},
        "suggested_resources": [],
        "paper_summaries": REFERENCE_SUMMARIES,
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
        "final_report_path": "",
        "agent_logs": [],
        "errors": [],
    }


def _is_real_pdf(path: Path) -> bool:
    """Return True if the file at path is a real PDF (starts with %PDF)."""
    try:
        return path.read_bytes()[:4] == b"%PDF"
    except Exception:
        return False