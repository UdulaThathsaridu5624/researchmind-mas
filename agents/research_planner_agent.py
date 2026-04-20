from __future__ import annotations

import json
import re
from typing import Any, Dict

from langchain_ollama import OllamaLLM

from logger import log_agent_event
from state import ResearchMindState
from tools.proposal_analyzer_tool import proposal_analyzer_tool

# Persona and constraints
SYSTEM_PROMPT = """You are an academic research advisor with deep expertise in \
computer science research methodology.
Your task is to analyze a research proposal and create a structured, realistic \
implementation plan.

Rules:
- Always respond with ONLY valid JSON. No markdown fences. No explanation outside JSON.
- The JSON must contain exactly these keys:
    "implementation_plan": a detailed paragraph (string) describing the overall plan,
    "timeline": an object with at least 4 milestone keys (e.g. "week_1_2", "week_3_4")
                where each value is {"tasks": [list of strings], "deliverable": "string"},
    "suggested_resources": a list of at least 5 academic resources, datasets, or tools
                           relevant to the research topic.
- Be specific, actionable, and realistic about timeframes.
- Do NOT include any text before or after the JSON object."""

_FALLBACK_TIMELINE: Dict[str, Any] = {
    "week_1_2": {
        "tasks": ["Review existing literature", "Set up development environment"],
        "deliverable": "Literature summary and environment ready",
    },
    "week_3_4": {
        "tasks": ["Design system architecture", "Define methodology"],
        "deliverable": "Architecture document",
    },
    "week_5_6": {
        "tasks": ["Implement core components", "Write unit tests"],
        "deliverable": "Working prototype",
    },
    "week_7_8": {
        "tasks": ["Evaluate system", "Write final report"],
        "deliverable": "Final draft submitted",
    },
}
_FALLBACK_RESOURCES = [
    "IEEE Xplore Digital Library",
    "ACM Digital Library",
    "arXiv.org",
    "Google Scholar",
    "Semantic Scholar",
]


def research_planner_agent(state: ResearchMindState) -> ResearchMindState:
    """
    LangGraph node - Research Planner Agent (Member 1).

    Reads the proposal PDF via proposal_analyzer_tool, then uses the
    configured Ollama model to generate:
      - A week-by-week implementation plan with milestones
      - A structured timeline (>= 4 entries)
      - A list of suggested academic resources

    Results are written to state keys:
        proposal_extracted, implementation_plan, timeline, suggested_resources.

    Errors are appended to state["errors"] rather than raised so the
    LangGraph conditional edge can decide whether to halt or continue.

    Args:
        state: Current ResearchMindState.

    Returns:
        Updated ResearchMindState.
    """
    model_name = state.get("model_name", "gemma4:e2b")
    state = log_agent_event(
        state,
        agent_name="research_planner_agent",
        event_type="START",
        input_data={
            "research_topic": state["research_topic"],
            "proposal_pdf_path": state["proposal_pdf_path"],
            "model_name": model_name,
        },
    )

    # Step 1: Extract proposal data
    try:
        extracted: Dict[str, Any] = proposal_analyzer_tool(state["proposal_pdf_path"])
        state = {**state, "proposal_extracted": extracted}
        state = log_agent_event(
            state,
            agent_name="research_planner_agent",
            event_type="TOOL_CALL",
            tool_calls=[
                {
                    "tool": "proposal_analyzer_tool",
                    "input": state["proposal_pdf_path"],
                    "output": {
                        "title": extracted.get("title"),
                        "keywords_count": len(extracted.get("keywords", [])),
                        "objectives_count": len(extracted.get("objectives", [])),
                    },
                }
            ],
        )
    except (FileNotFoundError, ValueError) as exc:
        error_msg = f"research_planner_agent: proposal_analyzer_tool failed - {exc}"
        state = {**state, "errors": list(state.get("errors", [])) + [error_msg]}
        state = log_agent_event(
            state,
            agent_name="research_planner_agent",
            event_type="ERROR",
            output_data={"error": error_msg},
        )
        return state

    # Step 2: Build LLM prompt
    user_prompt = f"""Research Topic: {state['research_topic']}

Proposal Details:
- Title: {extracted.get('title', 'N/A')}
- Objectives: {json.dumps(extracted.get('objectives', []))}
- Scope: {extracted.get('scope', 'N/A')}
- Methodology: {extracted.get('methodology', 'N/A')}
- Keywords: {json.dumps(extracted.get('keywords', []))}

Additional context from the proposal:
{extracted.get('raw_text', '')[:1500]}

Generate a complete research implementation plan as a JSON object."""

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    # Step 3: Call Ollama
    try:
        llm = OllamaLLM(model=model_name, temperature=0.3)
        raw_response: str = llm.invoke(full_prompt)
    except Exception as exc:
        error_msg = f"research_planner_agent: Ollama call failed - {exc}"
        state = {**state, "errors": list(state.get("errors", [])) + [error_msg]}
        state = log_agent_event(
            state,
            agent_name="research_planner_agent",
            event_type="ERROR",
            output_data={"error": error_msg},
        )
        return state

    # Step 4: Parse JSON with 3-tier fallback
    parsed = _parse_llm_json(raw_response)

    timeline = parsed.get("timeline", {})
    if len(timeline) < 4:
        timeline = _FALLBACK_TIMELINE

    suggested_resources = parsed.get("suggested_resources", [])
    if not suggested_resources:
        suggested_resources = _FALLBACK_RESOURCES

    state = {
        **state,
        "implementation_plan": parsed.get("implementation_plan", raw_response),
        "timeline": timeline,
        "suggested_resources": suggested_resources,
    }

    state = log_agent_event(
        state,
        agent_name="research_planner_agent",
        event_type="OUTPUT",
        output_data={
            "implementation_plan_length": len(state["implementation_plan"]),
            "timeline_milestones": len(state["timeline"]),
            "suggested_resources_count": len(state["suggested_resources"]),
        },
    )
    return state


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Parse LLM output to a dict using a 3-tier fallback strategy.

    Tier 1: Direct json.loads on the raw string.
    Tier 2: Extract the first {...} block via regex, then json.loads.
    Tier 3: Return a minimal structure with raw text as implementation_plan.

    Args:
        raw: Raw string output from the LLM.

    Returns:
        Parsed dict, guaranteed to be a dict (never raises).
    """
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "implementation_plan": raw.strip(),
        "timeline": {},
        "suggested_resources": [],
    }
