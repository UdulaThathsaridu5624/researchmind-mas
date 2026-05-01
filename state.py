from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import TypedDict


class ResearchMindState(TypedDict):
    """
    Shared state schema for the ResearchMind Multi-Agent System.
    This is the contract between all 4 agents - every field must be
    initialized before the graph is invoked (see main.py).
    """

    # Inputs
    research_topic: str
    proposal_pdf_path: str
    paper_pdf_paths: List[str]
    own_paper_path: str
    model_name: str

    # Agent 1: Research Planner outputs
    proposal_extracted: Dict[str, Any]
    implementation_plan: str
    timeline: Dict[str, Any]
    suggested_resources: List[str]

    # Agent 2: Literature Review outputs
    paper_summaries: List[Dict[str, Any]]
    citation_map: Dict[str, Any]
    core_themes: List[str]
    section_explanations: Dict[str, Any]
    literature_review_report: Dict[str, Any]

    # Agent 3: Gap and Hypothesis outputs
    identified_gaps: List[str]
    gap_frequency_scores: Dict[str, float]
    hypotheses: List[str]
    positioning_statement: str
    contradiction_pairs: List[Dict[str, Any]]

    # Agent 4: Paper Auditor outputs
    formatting_issues: List[str]
    missing_sections: List[str]
    plagiarism_score: float
    integrity_score: float
    audit_feedback: str
    final_report_path: str

    # Observability
    agent_logs: List[Dict[str, Any]]
    errors: List[str]
