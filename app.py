"""
ResearchMind MAS — Streamlit UI

Two run modes:
  Full Pipeline : user uploads a proposal PDF → all 4 agents run in sequence
                  and each responds as a chat message.
  Single Agent  : choose any one agent, provide its inputs, click Run —
                  prerequisite agents execute silently in the background.

Run:
    python -m streamlit run app.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import streamlit as st

from agents.gap_hypothesis_agent import gap_hypothesis_agent
from agents.literature_review_agent import literature_review_agent
from agents.paper_auditor_agent import paper_auditor_agent
from agents.research_planner_agent import research_planner_agent
from graph import build_graph
from state import ResearchMindState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = ["gemma4:e2b", "llama3:8b", "phi3", "qwen2.5:7b"]

AGENT_DISPLAY = {
    "research_planner":  "Agent 1 · Research Planner",
    "literature_review": "Agent 2 · Literature Review",
    "gap_hypothesis":    "Agent 3 · Gap & Hypothesis",
    "paper_auditor":     "Agent 4 · Paper Auditor",
}

AGENT_DESCRIPTIONS = {
    "research_planner": (
        "Reads the proposal PDF, extracts objectives/scope/methodology/citations, "
        "then generates a week-by-week implementation plan and suggests resources."
    ),
    "literature_review": (
        "Treats the proposal as the primary paper. Parses each literature PDF and "
        "compares it against the proposal — similarities, differences, gaps highlighted, "
        "and an LLM-written synthesis narrative."
    ),
    "gap_hypothesis": (
        "Reads the literature review outputs and identifies research gaps, contradiction "
        "pairs across papers, novel hypotheses, and a positioning statement for the proposal."
    ),
    "paper_auditor": (
        "Audits the proposal (or an uploaded own paper) for missing sections, "
        "formatting issues, plagiarism similarity against literature PDFs, and "
        "generates detailed written feedback aligned with the identified research gaps."
    ),
}

SINGLE_AGENT_OPTIONS = [
    "Agent 1 · Research Planner",
    "Agent 2 · Literature Review",
    "Agent 3 · Gap & Hypothesis",
    "Agent 4 · Paper Auditor",
]

AGENT_NODE_MAP = {
    "Agent 1 · Research Planner": "research_planner",
    "Agent 2 · Literature Review": "literature_review",
    "Agent 3 · Gap & Hypothesis":  "gap_hypothesis",
    "Agent 4 · Paper Auditor":     "paper_auditor",
}


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Shared state builder
# ---------------------------------------------------------------------------

def _empty_state(
    research_topic: str = "",
    proposal_pdf_path: str = "",
    paper_pdf_paths: Optional[List[str]] = None,
    own_paper_path: str = "",
    model_name: str = "gemma4:e2b",
) -> ResearchMindState:
    return ResearchMindState(
        research_topic=research_topic,
        proposal_pdf_path=proposal_pdf_path,
        paper_pdf_paths=paper_pdf_paths or [],
        own_paper_path=own_paper_path,
        model_name=model_name,
        proposal_extracted={},
        implementation_plan="",
        timeline={},
        suggested_resources=[],
        paper_summaries=[],
        citation_map={},
        core_themes=[],
        section_explanations={},
        literature_review_report={},
        identified_gaps=[],
        gap_frequency_scores={},
        hypotheses=[],
        positioning_statement="",
        contradiction_pairs=[],
        formatting_issues=[],
        missing_sections=[],
        plagiarism_score=0.0,
        integrity_score=0.0,
        audit_feedback="",
        final_report_path="outputs/reports/final_report.json",
        agent_logs=[],
        errors=[],
    )


# ---------------------------------------------------------------------------
# Temp-file helpers
# ---------------------------------------------------------------------------

def _save_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def _cleanup(paths: List[str]) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-agent renderers
# ---------------------------------------------------------------------------

def _render_research_planner(state: Dict[str, Any]) -> None:
    extracted = state.get("proposal_extracted", {})
    if extracted.get("title"):
        st.markdown(f"**Detected title:** {extracted['title']}")
    if extracted.get("keywords"):
        st.markdown("**Keywords:** " + ", ".join(extracted["keywords"]))

    refs = extracted.get("references", [])
    if refs:
        st.markdown(f"**References found in proposal:** {len(refs)}")

    if extracted.get("objectives"):
        with st.expander("Objectives", expanded=False):
            for obj in extracted["objectives"]:
                st.markdown(f"- {obj}")
    if extracted.get("scope"):
        with st.expander("Scope", expanded=False):
            st.markdown(extracted["scope"])
    if extracted.get("methodology"):
        with st.expander("Methodology", expanded=False):
            st.markdown(extracted["methodology"])

    plan = state.get("implementation_plan", "")
    if plan:
        with st.expander("Implementation Plan", expanded=True):
            st.markdown(plan)

    timeline = state.get("timeline", {})
    if timeline:
        with st.expander(f"Timeline — {len(timeline)} milestones", expanded=False):
            for milestone, details in timeline.items():
                label = milestone.replace("_", " ").title()
                if isinstance(details, dict):
                    tasks = details.get("tasks", [])
                    deliverable = details.get("deliverable", "")
                    st.markdown(f"**{label}**")
                    for t in tasks:
                        st.markdown(f"  - {t}")
                    if deliverable:
                        st.caption(f"Deliverable: {deliverable}")
                else:
                    st.markdown(f"**{label}:** {details}")

    resources = state.get("suggested_resources", [])
    if resources:
        with st.expander(f"Suggested Resources ({len(resources)})", expanded=False):
            for r in resources:
                st.markdown(f"- {r}")


def _render_literature_review(state: Dict[str, Any]) -> None:
    report = state.get("literature_review_report", {})
    primary_title = report.get("primary_paper_title", "")
    if primary_title:
        st.markdown(f"**Base paper:** {primary_title}")

    themes = state.get("core_themes", [])
    if themes:
        st.markdown("**Core themes:** " + ", ".join(themes))

    # LLM synthesis narrative
    synthesis = report.get("synthesis", {})
    if isinstance(synthesis, dict) and synthesis.get("narrative"):
        with st.expander("Literature Review Narrative", expanded=True):
            st.markdown(synthesis["narrative"])
            if synthesis.get("research_positioning"):
                st.markdown("**Positioning:** " + synthesis["research_positioning"])
            if synthesis.get("gaps_addressed"):
                st.markdown("**Gaps addressed:** " + synthesis["gaps_addressed"])
            if synthesis.get("novelty_assessment"):
                st.markdown("**Novelty:** " + synthesis["novelty_assessment"])

    # Primary paper analysis
    primary_analysis = report.get("primary_paper_analysis", {})
    if isinstance(primary_analysis, dict) and primary_analysis.get("research_problem"):
        with st.expander("Proposal Analysis", expanded=False):
            for key, val in primary_analysis.items():
                if val and not key.startswith("error"):
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {val}")

    # Per-paper comparisons
    comparisons = report.get("paper_comparisons", [])
    if comparisons:
        with st.expander(f"Paper Comparisons ({len(comparisons)})", expanded=False):
            for item in comparisons:
                title = item.get("paper_title", "Unknown")
                comp = item.get("comparison", {})
                rel = comp.get("relationship_type", "")
                badge = {"complementary": "🟢", "contradictory": "🔴",
                         "foundational": "🔵", "extension": "🟡"}.get(rel, "⚪")
                st.markdown(f"{badge} **{title}** — *{rel}*")
                if comp.get("similarities"):
                    st.caption("Similarities: " + comp["similarities"][:200])
                if comp.get("gaps_it_highlights"):
                    st.caption("Gaps highlighted: " + comp["gaps_it_highlights"][:200])
                st.divider()

    # Paper summaries
    summaries = state.get("paper_summaries", [])
    if summaries:
        with st.expander(f"Parsed Papers ({len(summaries)})", expanded=False):
            for i, paper in enumerate(summaries, 1):
                title = paper.get("title") or f"Paper {i}"
                st.markdown(f"**{i}. {title}**")
                abstract = paper.get("abstract", "")
                if abstract and "not found" not in abstract.lower():
                    st.markdown(f"> {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
                findings = paper.get("key_findings", [])
                if findings:
                    for f in findings[:3]:
                        st.markdown(f"  - {f}")
                refs = paper.get("citations", [])
                if refs:
                    st.caption(f"{len(refs)} references")
                st.divider()


def _render_gap_hypothesis(state: Dict[str, Any]) -> None:
    positioning = state.get("positioning_statement", "")
    if positioning:
        st.info(positioning)

    gaps = state.get("identified_gaps", [])
    if gaps:
        with st.expander(f"Research Gaps ({len(gaps)})", expanded=True):
            for gap in gaps:
                st.markdown(f"- {gap}")

    scores = state.get("gap_frequency_scores", {})
    if scores:
        with st.expander("Gap Frequency Scores", expanded=False):
            for gap_key, score in list(scores.items())[:8]:
                label = gap_key.replace("_", " ").title()
                st.progress(float(score), text=f"{label}: {score:.2f}")

    hypotheses = state.get("hypotheses", [])
    if hypotheses:
        with st.expander(f"Research Hypotheses ({len(hypotheses)})", expanded=True):
            for h in hypotheses:
                st.markdown(f"- {h}")

    contradictions = state.get("contradiction_pairs", [])
    if contradictions:
        with st.expander(f"Contradictions Detected ({len(contradictions)})", expanded=False):
            for pair in contradictions:
                aspect = pair.get("aspect", "unknown").title()
                st.markdown(f"**{aspect}**")
                st.markdown(f"- *{pair.get('paper_a', '')}*: {pair.get('claim_a', '')}")
                st.markdown(f"- *{pair.get('paper_b', '')}*: {pair.get('claim_b', '')}")
                st.divider()


def _render_paper_auditor(state: Dict[str, Any]) -> None:
    feedback = state.get("audit_feedback", "")
    if feedback == "No paper was provided for auditing." or (
        not state.get("own_paper_path") and not state.get("proposal_pdf_path")
    ):
        st.caption("No paper was provided for auditing — skipped.")
        return

    plagiarism = state.get("plagiarism_score", 0.0)
    integrity = state.get("integrity_score", 0.0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Plagiarism Score", f"{plagiarism:.2f}",
                  help="0.0 = original, 1.0 = identical to a reference paper")
    with col2:
        badge = "GOOD" if integrity >= 0.8 else "NEEDS WORK" if integrity >= 0.6 else "POOR"
        st.metric("Integrity Score", f"{integrity:.2f} — {badge}")

    missing = state.get("missing_sections", [])
    if missing:
        st.warning(f"**Missing sections:** {', '.join(missing)}")
    else:
        st.success("All required sections present.")

    formatting = state.get("formatting_issues", [])
    if formatting:
        with st.expander(f"Formatting Issues ({len(formatting)})", expanded=False):
            for issue in formatting:
                st.markdown(f"- {issue}")

    if feedback:
        with st.expander("Detailed Audit Feedback", expanded=True):
            st.markdown(feedback)

    report_path = state.get("final_report_path", "")
    if report_path and Path(report_path).exists():
        with open(report_path, encoding="utf-8") as f:
            md_content = f.read()
        st.download_button(
            "Download Audit Report (Markdown)",
            data=md_content,
            file_name="audit_report.md",
            mime="text/markdown",
        )


AGENT_RENDERERS = {
    "research_planner":  _render_research_planner,
    "literature_review": _render_literature_review,
    "gap_hypothesis":    _render_gap_hypothesis,
    "paper_auditor":     _render_paper_auditor,
}


# ---------------------------------------------------------------------------
# Single-agent runner (executes prerequisites silently)
# ---------------------------------------------------------------------------

def _run_single_agent(
    agent_label: str,
    research_topic: str,
    proposal_path: str,
    paper_paths: List[str],
    own_paper_path: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Run one agent (and any silent prerequisites) and return the final state dict.
    Prerequisites for Agents 2/3 run with a Streamlit spinner but their outputs
    are not rendered — only the selected agent's output is shown.
    """
    state: Dict[str, Any] = dict(_empty_state(
        research_topic=research_topic,
        proposal_pdf_path=proposal_path,
        paper_pdf_paths=paper_paths,
        own_paper_path=own_paper_path,
        model_name=model_name,
    ))

    def _fill_topic(s: Dict[str, Any]) -> Dict[str, Any]:
        """After Agent 1, auto-populate research_topic from extracted title."""
        if not s.get("research_topic"):
            title = s.get("proposal_extracted", {}).get("title", "")
            if title:
                return {**s, "research_topic": title}
        return s

    if agent_label == "Agent 1 · Research Planner":
        with st.spinner("Agent 1 · Research Planner running…"):
            state = research_planner_agent(state)

    elif agent_label == "Agent 2 · Literature Review":
        with st.spinner("Agent 1 · Research Planner (prerequisite)…"):
            state = _fill_topic(research_planner_agent(state))
        with st.spinner("Agent 2 · Literature Review running…"):
            state = literature_review_agent(state)

    elif agent_label == "Agent 3 · Gap & Hypothesis":
        with st.spinner("Agent 1 · Research Planner (prerequisite)…"):
            state = _fill_topic(research_planner_agent(state))
        with st.spinner("Agent 2 · Literature Review (prerequisite)…"):
            state = literature_review_agent(state)
        with st.spinner("Agent 3 · Gap & Hypothesis running…"):
            state = gap_hypothesis_agent(state)

    elif agent_label == "Agent 4 · Paper Auditor":
        with st.spinner("Agent 4 · Paper Auditor running…"):
            state = paper_auditor_agent(state)

    return state


# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMind MAS",
    page_icon="🔬",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if "single_node" not in st.session_state:
    st.session_state.single_node = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 ResearchMind")
    st.caption("Multi-Agent Research Assistant")
    st.divider()

    ollama_ok = _ollama_running()
    if ollama_ok:
        st.success("Ollama is running")
    else:
        st.error("Ollama offline — run `ollama serve`")

    st.divider()

    run_mode = st.radio(
        "Run Mode",
        ["Full Pipeline", "Single Agent"],
        horizontal=True,
    )

    st.divider()

    # ── Shared file uploaders ──────────────────────────────────────────────
    proposal_file = st.file_uploader(
        "Proposal PDF *(required)*",
        type=["pdf"],
        help="Primary document — analyzed by all agents.",
    )

    # Literature PDFs: shown for Full Pipeline and agents 2/3/4
    show_lit = run_mode == "Full Pipeline" or (
        run_mode == "Single Agent"  # always show; agent 1 just ignores them
    )
    paper_files = st.file_uploader(
        "Literature PDFs *(optional)*",
        type=["pdf"],
        accept_multiple_files=True,
        help="Reference papers compared against the proposal.",
    ) if show_lit else []

    # Own paper: Full Pipeline or Agent 4 individual
    show_own = run_mode == "Full Pipeline" or (
        run_mode == "Single Agent"
    )
    own_paper_file = st.file_uploader(
        "Own Paper PDF *(optional — Agent 4 audits this; defaults to Proposal)*",
        type=["pdf"],
        help="If omitted, Agent 4 audits the proposal PDF instead.",
    ) if show_own else None

    st.divider()

    # Agent selector — only in Single Agent mode
    selected_agent_label: str = ""
    if run_mode == "Single Agent":
        selected_agent_label = st.selectbox(
            "Select Agent",
            SINGLE_AGENT_OPTIONS,
            index=0,
        )
        node_key = AGENT_NODE_MAP[selected_agent_label]
        st.caption(AGENT_DESCRIPTIONS[node_key])
        st.divider()
        st.caption("Research topic will be extracted from the proposal PDF.")

    model_name = st.selectbox("Ollama Model", MODELS, index=0)

    # ── Single Agent run button ────────────────────────────────────────────
    run_single = False
    if run_mode == "Single Agent":
        st.divider()
        run_single = st.button(
            f"▶ Run {selected_agent_label}",
            disabled=not ollama_ok or proposal_file is None,
            use_container_width=True,
        )
        if proposal_file is None:
            st.caption("Upload a Proposal PDF to enable.")

    # ── Full Pipeline download button (post-run) ───────────────────────────
    if run_mode == "Full Pipeline" and st.session_state.final_state:
        st.divider()
        report_json = json.dumps(
            dict(st.session_state.final_state), indent=2, default=str
        )
        st.download_button(
            "⬇ Download Final Report (JSON)",
            data=report_json,
            file_name="final_report.json",
            mime="application/json",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main area — Full Pipeline mode
# ---------------------------------------------------------------------------

if run_mode == "Full Pipeline":
    st.title("ResearchMind MAS — Full Pipeline")
    st.caption(
        "Upload your Proposal PDF (and optionally literature PDFs + own paper) in the "
        "sidebar, then run the pipeline. The research topic is extracted from the proposal."
    )

    # Re-render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"### {AGENT_DISPLAY.get(msg['node'], msg['node'])}")
                AGENT_RENDERERS[msg["node"]](msg["state"])

    run_full = st.button(
        "▶ Run Full Pipeline",
        disabled=not ollama_ok or proposal_file is None,
        use_container_width=True,
    )
    if proposal_file is None:
        st.caption("Upload a Proposal PDF to enable.")

    if run_full:
        if proposal_file is None:
            st.warning("Please upload a Proposal PDF in the sidebar first.")
            st.stop()

        prompt = "Run full pipeline using the topic extracted from the proposal PDF."
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        tmp_paths: List[str] = []
        try:
            proposal_path = _save_upload(proposal_file)
            tmp_paths.append(proposal_path)
            lit_paths = []
            for f in (paper_files or []):
                p = _save_upload(f)
                lit_paths.append(p)
                tmp_paths.append(p)
            own_path = ""
            if own_paper_file:
                own_path = _save_upload(own_paper_file)
                tmp_paths.append(own_path)

            initial_state = _empty_state(
                research_topic="",
                proposal_pdf_path=proposal_path,
                paper_pdf_paths=lit_paths,
                own_paper_path=own_path,
                model_name=model_name,
            )

            graph = build_graph()
            merged: Dict[str, Any] = dict(initial_state)

            for event in graph.stream(initial_state):
                for node_name, node_state in event.items():
                    merged.update(node_state)
                    # After Agent 1 runs, fill research_topic from the extracted
                    # proposal title so Agents 2–4 get a real topic string.
                    if node_name == "research_planner" and not merged.get("research_topic"):
                        extracted_title = node_state.get("proposal_extracted", {}).get("title", "")
                        if extracted_title:
                            merged["research_topic"] = extracted_title
                    snapshot = dict(merged)

                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(f"### {AGENT_DISPLAY.get(node_name, node_name)}")
                        AGENT_RENDERERS.get(node_name, lambda s: st.json(s))(snapshot)

                    st.session_state.messages.append(
                        {"role": "assistant", "node": node_name, "state": snapshot, "content": ""}
                    )

                    for err in node_state.get("errors", []):
                        st.error(err)

            st.session_state.final_state = merged

            report_path = Path("outputs/reports/final_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(merged, fh, indent=2, default=str)

            st.rerun()

        finally:
            _cleanup(tmp_paths)


# ---------------------------------------------------------------------------
# Main area — Single Agent mode
# ---------------------------------------------------------------------------

else:
    node_key = AGENT_NODE_MAP.get(selected_agent_label, "research_planner")

    st.title(f"ResearchMind MAS — {selected_agent_label}")
    st.markdown(f"_{AGENT_DESCRIPTIONS[node_key]}_")
    st.divider()

    # Show prerequisite notice
    prereqs = {
        "Agent 2 · Literature Review": ["Agent 1"],
        "Agent 3 · Gap & Hypothesis":  ["Agent 1", "Agent 2"],
    }
    if selected_agent_label in prereqs:
        p = prereqs[selected_agent_label]
        st.info(
            f"**{' and '.join(p)} will run silently** in the background to populate "
            "the required state before this agent executes."
        )

    # Show previous result if agent hasn't changed
    if (
        st.session_state.single_result is not None
        and st.session_state.single_node == node_key
    ):
        result_state = st.session_state.single_result
        errors = result_state.get("errors", [])
        if errors:
            for err in errors:
                st.error(err)
        AGENT_RENDERERS[node_key](result_state)

        # Download for single-agent run
        report_json = json.dumps(result_state, indent=2, default=str)
        st.divider()
        st.download_button(
            "⬇ Download Agent Output (JSON)",
            data=report_json,
            file_name=f"{node_key}_output.json",
            mime="application/json",
        )
    else:
        st.markdown(
            "Configure inputs in the sidebar, then click **▶ Run** to execute this agent."
        )

    # Handle Run button press
    if run_single:
        tmp_paths = []
        try:
            proposal_path = _save_upload(proposal_file)
            tmp_paths.append(proposal_path)
            lit_paths = []
            for f in (paper_files or []):
                p = _save_upload(f)
                lit_paths.append(p)
                tmp_paths.append(p)
            own_path = ""
            if own_paper_file:
                own_path = _save_upload(own_paper_file)
                tmp_paths.append(own_path)

            result = _run_single_agent(
                agent_label=selected_agent_label,
                research_topic="",
                proposal_path=proposal_path,
                paper_paths=lit_paths,
                own_paper_path=own_path,
                model_name=model_name,
            )

            st.session_state.single_result = result
            st.session_state.single_node = node_key
            st.rerun()

        finally:
            _cleanup(tmp_paths)
