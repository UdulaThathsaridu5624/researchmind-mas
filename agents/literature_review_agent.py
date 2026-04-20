from __future__ import annotations

from logger import log_agent_event
from state import ResearchMindState


def literature_review_agent(state: ResearchMindState) -> ResearchMindState:
    """
    STUB — Literature Review Agent (Member 2: Tharindu).

    TODO (Member 2):
        - Read paper_pdf_paths from state
        - Summarize each paper using paper_parser_tool
        - Write to: paper_summaries, citation_map, core_themes, section_explanations

    Currently passes state through unchanged so the graph compiles and runs.
    """
    state = log_agent_event(
        state,
        agent_name="literature_review_agent",
        event_type="START",
        input_data={"stub": True, "paper_count": len(state.get("paper_pdf_paths", []))},
    )
    state = log_agent_event(
        state,
        agent_name="literature_review_agent",
        event_type="OUTPUT",
        output_data={"stub": True, "message": "Not yet implemented — Member 2"},
    )
    return state
