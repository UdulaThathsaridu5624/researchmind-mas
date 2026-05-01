from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from state import ResearchMindState

LOGS_DIR = Path("outputs/logs")

# Session file path is set once on the first call so all events in a single
# run share the same JSONL file.
_session_file: Optional[Path] = None


def _get_session_file() -> Path:
    """Return the JSONL log file for this process run, creating it on first call."""
    global _session_file
    if _session_file is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _session_file = LOGS_DIR / f"session_{ts}.jsonl"
    return _session_file


def log_agent_event(
    state: ResearchMindState,
    agent_name: str,
    event_type: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> ResearchMindState:
    """
    Record a structured agent event to both in-memory state and a JSONL file.

    Args:
        state:       Current ResearchMindState (not mutated — a new dict is returned).
        agent_name:  Name of the calling agent (e.g. "research_planner_agent").
        event_type:  One of "START", "TOOL_CALL", "OUTPUT", "ERROR".
        input_data:  Optional dict of inputs relevant to this event.
        output_data: Optional dict of outputs produced by this event.
        tool_calls:  Optional list of tool call records
                     (each a dict with "tool", "input", "output").

    Returns:
        Updated ResearchMindState with the new log entry appended to agent_logs.
    """
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_name": agent_name,
        "event_type": event_type,
        "input_data": input_data or {},
        "output_data": output_data or {},
        "tool_calls": tool_calls or [],
    }

    # Write to JSONL file
    try:
        with open(_get_session_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # never let logging crash the agent pipeline

    # Return updated state (immutable pattern for LangGraph)
    updated_logs: List[Dict[str, Any]] = list(state.get("agent_logs", [])) + [record]
    return {**state, "agent_logs": updated_logs}
