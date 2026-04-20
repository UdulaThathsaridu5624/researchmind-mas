from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.gap_hypothesis_agent import gap_hypothesis_agent
from agents.literature_review_agent import literature_review_agent
from agents.paper_auditor_agent import paper_auditor_agent
from agents.research_planner_agent import research_planner_agent
from state import ResearchMindState


def _has_critical_error(state: ResearchMindState) -> str:
    """
    Conditional edge function.

    Returns "error"    if state["errors"] is non-empty (halts the pipeline).
    Returns "continue" otherwise (advances to the next agent).
    """
    if state.get("errors"):
        return "error"
    return "continue"


def build_graph() -> StateGraph:
    """
    Build and compile the ResearchMind MAS LangGraph StateGraph.

    Pipeline:
        research_planner → literature_review → gap_hypothesis → paper_auditor → END

    After each agent a conditional edge checks state["errors"]. If an error
    was recorded the graph short-circuits directly to END, preserving all
    state collected so far for inspection.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream().
    """
    graph: StateGraph = StateGraph(ResearchMindState)

    # Register agent nodes
    graph.add_node("research_planner", research_planner_agent)
    graph.add_node("literature_review", literature_review_agent)
    graph.add_node("gap_hypothesis", gap_hypothesis_agent)
    graph.add_node("paper_auditor", paper_auditor_agent)

    # Entry point
    graph.set_entry_point("research_planner")

    # Conditional edges — error short-circuit after each agent
    graph.add_conditional_edges(
        "research_planner",
        _has_critical_error,
        {"continue": "literature_review", "error": END},
    )
    graph.add_conditional_edges(
        "literature_review",
        _has_critical_error,
        {"continue": "gap_hypothesis", "error": END},
    )
    graph.add_conditional_edges(
        "gap_hypothesis",
        _has_critical_error,
        {"continue": "paper_auditor", "error": END},
    )
    graph.add_edge("paper_auditor", END)

    return graph.compile()
