from __future__ import annotations

from logger import log_agent_event
from state import ResearchMindState
from tools.gap_analyzer_tool import gap_analyzer_tool


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
            research_topic=state.get("research_topic", "the given topic"),
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
