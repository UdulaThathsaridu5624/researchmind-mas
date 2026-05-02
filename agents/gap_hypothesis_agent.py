from __future__ import annotations

import re

from logger import log_agent_event
from state import ResearchMindState
from tools.gap_analyzer_tool import gap_analyzer_tool


_GENERIC_TOPIC_PREFIXES = (
    "project proposal",
    "proposal",
    "research proposal",
    "thesis proposal",
    "final report",
    "research paper",
    "paper",
    "report",
    "manuscript",
)


def _clean_topic_text(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    text = text.replace("_", " ").replace("/", " ").replace("\\", " ")
    text = re.sub(r"\.(pdf|docx?|pptx?|txt)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" \t\n\r:;,-")

    for separator in (" | ", " - ", " -- "):
        if separator in text:
            parts = [part.strip() for part in text.split(separator) if part.strip()]
            if parts:
                text = parts[-1]
                break

    match = re.match(
        r"^(?:" + "|".join(re.escape(prefix) for prefix in _GENERIC_TOPIC_PREFIXES) + r")\s*[:\-–—|]\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()

    if ":" in text:
        head, tail = [part.strip() for part in text.split(":", 1)]
        if head and tail and head.lower() in _GENERIC_TOPIC_PREFIXES:
            text = tail

    return re.sub(r"\s+", " ", text).strip(" \t\n\r:;,-")


def _resolve_gap_topic(state: ResearchMindState) -> str:
    proposal = state.get("proposal_extracted", {}) or {}
    proposal_title = _clean_topic_text(proposal.get("title", ""))
    if proposal_title:
        return proposal_title

    keywords = proposal.get("keywords", [])
    if isinstance(keywords, list):
        keyword_text = _clean_topic_text(", ".join(str(keyword) for keyword in keywords[:5] if str(keyword).strip()))
        if keyword_text:
            return keyword_text

    return _clean_topic_text(state.get("research_topic", "")) or "the given topic"


def gap_hypothesis_agent(state: ResearchMindState) -> ResearchMindState:
    """
    LangGraph node - Research Gap & Hypothesis Agent (Member 3).

    Reads literature outputs and produces:
      - identified_gaps
      - gap_frequency_scores
      - hypotheses
      - positioning_statement
      - contradiction_pairs

    Errors are appended to state["errors"] instead of being raised.
    """
    paper_summaries = state.get("paper_summaries", [])
    citation_map = state.get("citation_map", {})
    core_themes = state.get("core_themes", [])
    resolved_topic = _resolve_gap_topic(state)

    state = log_agent_event(
        state,
        agent_name="gap_hypothesis_agent",
        event_type="START",
        input_data={
            "paper_summaries_count": len(paper_summaries),
            "citation_map_keys": len(citation_map) if isinstance(citation_map, dict) else 0,
            "themes_count": len(core_themes),
        },
    )

    try:
        analysis = gap_analyzer_tool(
            paper_summaries=paper_summaries,
            citation_map=citation_map,
            core_themes=core_themes,
            research_topic=resolved_topic,
        )
        state = log_agent_event(
            state,
            agent_name="gap_hypothesis_agent",
            event_type="TOOL_CALL",
            tool_calls=[
                {
                    "tool": "gap_analyzer_tool",
                    "input": {
                        "paper_summaries_count": len(paper_summaries),
                        "citation_map_keys": len(citation_map) if isinstance(citation_map, dict) else 0,
                        "themes_count": len(core_themes),
                        "research_topic": resolved_topic,
                    },
                    "output": {
                        "identified_gaps_count": len(analysis.get("identified_gaps", [])),
                        "hypotheses_count": len(analysis.get("hypotheses", [])),
                        "contradiction_pairs_count": len(analysis.get("contradiction_pairs", [])),
                    },
                }
            ],
        )
    except Exception as exc:
        error_msg = f"gap_hypothesis_agent: gap_analyzer_tool failed - {exc}"
        state = {**state, "errors": list(state.get("errors", [])) + [error_msg]}
        state = log_agent_event(
            state,
            agent_name="gap_hypothesis_agent",
            event_type="ERROR",
            output_data={"error": error_msg},
        )
        return state

    state = {
        **state,
        "identified_gaps": analysis.get("identified_gaps", []),
        "gap_frequency_scores": analysis.get("gap_frequency_scores", {}),
        "hypotheses": analysis.get("hypotheses", []),
        "positioning_statement": analysis.get("positioning_statement", ""),
        "contradiction_pairs": analysis.get("contradiction_pairs", []),
    }

    state = log_agent_event(
        state,
        agent_name="gap_hypothesis_agent",
        event_type="OUTPUT",
        output_data={
            "identified_gaps_count": len(state.get("identified_gaps", [])),
            "gap_frequency_scores_count": len(state.get("gap_frequency_scores", {})),
            "hypotheses_count": len(state.get("hypotheses", [])),
            "positioning_statement_length": len(state.get("positioning_statement", "")),
            "contradiction_pairs_count": len(state.get("contradiction_pairs", [])),
        },
    )
    return state
