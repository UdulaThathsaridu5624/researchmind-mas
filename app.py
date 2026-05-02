"""
ResearchMind MAS — Streamlit UI

Two modes:
  Full Pipeline : all 4 agents run sequentially. Each shows a live
                  "generating…" bubble (threaded) with a stop button.
  Single Agent  : pick one agent, run it independently.

Run:
    python -m streamlit run app.py
"""
from __future__ import annotations

import json
import os
import queue as queue_module
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
import streamlit as st
from fpdf import FPDF

from agents.gap_hypothesis_agent import gap_hypothesis_agent
from agents.literature_review_agent import literature_review_agent
from agents.paper_auditor_agent import paper_auditor_agent
from agents.research_planner_agent import research_planner_agent
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

AGENT_ORDER = [
    "research_planner",
    "literature_review",
    "gap_hypothesis",
    "paper_auditor",
]

AGENT_FUNCS = {
    "research_planner":  research_planner_agent,
    "literature_review": literature_review_agent,
    "gap_hypothesis":    gap_hypothesis_agent,
    "paper_auditor":     paper_auditor_agent,
}

AGENT_DESCRIPTIONS = {
    "research_planner": (
        "Extracts objectives, scope, methodology and citations from the proposal PDF, "
        "then generates a week-by-week implementation plan and resource suggestions."
    ),
    "literature_review": (
        "Treats the proposal as the primary paper. Parses each literature PDF, "
        "compares it against the proposal, and produces an LLM-written synthesis."
    ),
    "gap_hypothesis": (
        "Identifies research gaps, contradiction pairs across papers, novel "
        "hypotheses, and a positioning statement — all deterministic, no LLM needed."
    ),
    "paper_auditor": (
        "Audits the proposal (or uploaded own paper) for missing sections, "
        "plagiarism similarity, and citation issues, then writes detailed feedback."
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
# Helpers
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


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


def _safe(text: str) -> str:
    """Encode text to latin-1 safely for fpdf2 core fonts."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _make_pdf(title: str, sections: List[tuple]) -> bytes:
    """
    Build a PDF from a list of (heading, body) tuples.
    Returns raw PDF bytes suitable for st.download_button.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cover title
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _safe(title), align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Generated by ResearchMind MAS", ln=True, align="C")
    pdf.ln(8)

    for heading, body in sections:
        if not body:
            continue
        # Section heading
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(30, 60, 100)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, _safe(f"  {heading}"), ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)

        # Body — split into paragraphs so multi_cell handles long text
        for para in re.split(r'\n{2,}|\n', str(body)):
            para = para.strip()
            if para:
                pdf.multi_cell(0, 5, _safe(para))
                pdf.ln(1)
        pdf.ln(4)

    return bytes(pdf.output())


def _state_to_pdf(state: Dict[str, Any]) -> bytes:
    """Convert the full pipeline final state into a structured PDF report."""
    topic = state.get("research_topic", "Research Report")
    extracted = state.get("proposal_extracted", {})
    report = state.get("literature_review_report", {})
    synthesis = report.get("synthesis", {}) if isinstance(report, dict) else {}

    sections: List[tuple] = [
        # Agent 1
        ("Agent 1 — Research Planner", ""),
        ("  Detected Title", extracted.get("title", "")),
        ("  Objectives", "\n".join(extracted.get("objectives", []))),
        ("  Scope", extracted.get("scope", "")),
        ("  Methodology", extracted.get("methodology", "")),
        ("  Implementation Plan", state.get("implementation_plan", "")),
        ("  Suggested Resources", "\n".join(state.get("suggested_resources", []))),
        # Agent 2
        ("Agent 2 — Literature Review", ""),
        ("  Narrative", synthesis.get("narrative", "")),
        ("  Research Positioning", synthesis.get("research_positioning", "")),
        ("  Gaps Addressed", synthesis.get("gaps_addressed", "")),
        ("  Novelty Assessment", synthesis.get("novelty_assessment", "")),
        # Agent 3
        ("Agent 3 — Gap & Hypothesis", ""),
        ("  Positioning Statement", state.get("positioning_statement", "")),
        ("  Identified Gaps", "\n".join(state.get("identified_gaps", []))),
        ("  Hypotheses", "\n".join(state.get("hypotheses", []))),
        # Agent 4
        ("Agent 4 — Paper Auditor", ""),
        ("  Audit Feedback", state.get("audit_feedback", "")),
        ("  Missing Sections", ", ".join(state.get("missing_sections", [])) or "None"),
        ("  Plagiarism Score", str(round(state.get("plagiarism_score", 0.0), 2))),
        ("  Integrity Score", str(round(state.get("integrity_score", 0.0), 2))),
    ]
    return _make_pdf(topic, sections)


def _md_to_pdf(md_text: str, title: str = "Report") -> bytes:
    """Convert a Markdown string to a PDF (best-effort, strips markdown syntax)."""
    # Strip markdown headings/bold/bullets for plain text rendering
    plain = re.sub(r'^#{1,6}\s*', '', md_text, flags=re.MULTILINE)
    plain = re.sub(r'\*\*(.*?)\*\*', r'\1', plain)
    plain = re.sub(r'\*(.*?)\*', r'\1', plain)
    plain = re.sub(r'`([^`]*)`', r'\1', plain)
    plain = re.sub(r'^\s*[-*]\s+', '• ', plain, flags=re.MULTILINE)
    plain = re.sub(r'^\|.*\|.*$', '', plain, flags=re.MULTILINE)  # strip tables
    plain = re.sub(r'-{3,}', '', plain)
    return _make_pdf(title, [("", plain)])


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
        st.caption(f"{len(refs)} references found in proposal")

    if extracted.get("objectives"):
        with st.expander("Objectives", expanded=False):
            for i, obj in enumerate(extracted["objectives"]):
                # Strip leading section numbers like "2." "2.1" "2. 1"
                clean = re.sub(r'^\d+[\.\s]+\d*\s*', '', obj.strip())
                st.markdown(clean or obj)
                if i < len(extracted["objectives"]) - 1:
                    st.divider()
    if extracted.get("scope"):
        with st.expander("Scope", expanded=False):
            scope = extracted["scope"]
            for para in re.split(r'\n{2,}', scope):
                if para.strip():
                    st.markdown(para.strip())
    if extracted.get("methodology"):
        with st.expander("Methodology", expanded=False):
            methodology = extracted["methodology"]
            paragraphs = [p.strip() for p in re.split(r'\n{2,}', methodology) if p.strip()]
            if len(paragraphs) > 1:
                for para in paragraphs:
                    st.markdown(para)
            else:
                # No paragraph breaks — split on sentence boundaries every ~3 sentences
                sentences = re.split(r'(?<=[.!?])\s+', methodology)
                chunk, chunks = [], []
                for j, s in enumerate(sentences):
                    chunk.append(s)
                    if len(chunk) >= 3:
                        chunks.append(' '.join(chunk))
                        chunk = []
                if chunk:
                    chunks.append(' '.join(chunk))
                for para in chunks:
                    st.markdown(para)

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
                    st.markdown(f"**{label}**")
                    for t in details.get("tasks", []):
                        st.markdown(f"  - {t}")
                    if details.get("deliverable"):
                        st.caption(f"Deliverable: {details['deliverable']}")
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

    primary_analysis = report.get("primary_paper_analysis", {})
    if isinstance(primary_analysis, dict) and primary_analysis.get("research_problem"):
        with st.expander("Proposal Analysis", expanded=False):
            for key, val in primary_analysis.items():
                if val and not key.startswith("error"):
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {val}")

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

    summaries = state.get("paper_summaries", [])
    if summaries:
        with st.expander(f"Parsed Papers ({len(summaries)})", expanded=False):
            for i, paper in enumerate(summaries, 1):
                title = paper.get("title") or f"Paper {i}"
                st.markdown(f"**{i}. {title}**")
                abstract = paper.get("abstract", "")
                if abstract and "not found" not in abstract.lower():
                    st.markdown(f"> {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
                for f in paper.get("key_findings", [])[:3]:
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
                st.progress(float(score), text=f"{gap_key.replace('_', ' ').title()}: {score:.2f}")

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
    if feedback == "No paper was provided for auditing.":
        st.caption("No paper provided — skipped.")
        return

    plagiarism = state.get("plagiarism_score", 0.0)
    integrity = state.get("integrity_score", 0.0)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Plagiarism Score", f"{plagiarism:.2f}",
                  help="0.0 = original · 1.0 = identical to a reference paper")
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
            "⬇ Download Audit Report (PDF)",
            data=_md_to_pdf(md_content, "Paper Audit Report"),
            file_name="audit_report.pdf",
            mime="application/pdf",
        )


AGENT_RENDERERS = {
    "research_planner":  _render_research_planner,
    "literature_review": _render_literature_review,
    "gap_hypothesis":    _render_gap_hypothesis,
    "paper_auditor":     _render_paper_auditor,
}


# ---------------------------------------------------------------------------
# Background pipeline worker
# ---------------------------------------------------------------------------

def _agent_thread(
    node_name: str,
    state: Dict[str, Any],
    result_q: "queue_module.Queue[tuple]",
    stop_event: threading.Event,
) -> None:
    """Run one agent in a background thread. Result goes into result_q."""
    if stop_event.is_set():
        result_q.put(("stopped", None))
        return
    try:
        result = AGENT_FUNCS[node_name](state)
        result_q.put(("success", result))
    except Exception as exc:
        result_q.put(("error", str(exc)))


def _start_agent(step: int) -> None:
    """Kick off the background thread for pipeline step `step`."""
    node_name = AGENT_ORDER[step]
    result_q: queue_module.Queue = queue_module.Queue(maxsize=1)
    t = threading.Thread(
        target=_agent_thread,
        args=(
            node_name,
            dict(st.session_state.pipeline_state),
            result_q,
            st.session_state.pipeline_stop,
        ),
        daemon=True,
    )
    t.start()
    st.session_state.pipeline_result_q = result_q
    st.session_state.pipeline_running = True


# ---------------------------------------------------------------------------
# Single-agent runner
# ---------------------------------------------------------------------------

def _run_single_agent(
    agent_label: str,
    proposal_path: str,
    paper_paths: List[str],
    own_paper_path: str,
    model_name: str,
) -> Dict[str, Any]:
    state: Dict[str, Any] = dict(_empty_state(
        proposal_pdf_path=proposal_path,
        paper_pdf_paths=paper_paths,
        own_paper_path=own_paper_path,
        model_name=model_name,
    ))

    def _fill_topic(s: Dict[str, Any]) -> Dict[str, Any]:
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
# Page config + session state
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMind MAS",
    page_icon="🔬",
    layout="wide",
)

_SS_DEFAULTS: Dict[str, Any] = {
    "messages":         [],
    "final_state":      None,
    "single_result":    None,
    "single_node":      None,
    # Full-pipeline state machine
    "pipeline_step":    -1,       # -1 = idle, 0-3 = running agent N
    "pipeline_state":   None,     # current merged ResearchMindState dict
    "pipeline_running": False,    # True while background thread is live
    "pipeline_result_q": None,    # Queue for thread → main thread
    "pipeline_stop":     None,     # threading.Event
    "pipeline_stopping": False,   # True once Stop clicked, before thread ends
    "pipeline_topic":    "",      # display string for user bubble
    "tmp_paths":        [],       # temp files to clean up after run
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# ① Drain pipeline result queue (runs at the TOP of every script execution)
# ---------------------------------------------------------------------------

if st.session_state.pipeline_running:
    q = st.session_state.pipeline_result_q
    try:
        msg_type, data = q.get_nowait()
        st.session_state.pipeline_running = False

        if msg_type == "success":
            merged = dict(data)
            step = st.session_state.pipeline_step

            # Auto-populate research_topic from extracted title (Agent 1 only)
            if step == 0:
                title = merged.get("proposal_extracted", {}).get("title", "")
                if title:
                    merged["research_topic"] = title
                    st.session_state.pipeline_topic = title
                    # Update the user bubble so it shows the real title permanently
                    for m in st.session_state.messages:
                        if m["role"] == "user":
                            m["content"] = title
                            break

            st.session_state.pipeline_state = merged

            node_name = AGENT_ORDER[step]
            st.session_state.messages.append({
                "role": "assistant",
                "node": node_name,
                "state": dict(merged),
            })

            errors = merged.get("errors", [])
            stopped = st.session_state.pipeline_stop.is_set()
            last_step = (step == len(AGENT_ORDER) - 1)

            if errors or stopped or last_step:
                # Pipeline finished (naturally, by error, or by stop)
                st.session_state.pipeline_step = -1
                st.session_state.pipeline_stopping = False
                st.session_state.final_state = merged
                _cleanup(st.session_state.tmp_paths)
                st.session_state.tmp_paths = []
                if stopped:
                    st.toast("Pipeline stopped.", icon="⏹")
            else:
                # Advance to next agent immediately
                next_step = step + 1
                st.session_state.pipeline_step = next_step
                _start_agent(next_step)

        elif msg_type == "stopped":
            st.session_state.pipeline_step = -1
            st.session_state.pipeline_stopping = False
            _cleanup(st.session_state.tmp_paths)
            st.session_state.tmp_paths = []

        elif msg_type == "error":
            st.session_state.pipeline_step = -1
            st.session_state.pipeline_running = False
            st.session_state.pipeline_stopping = False
            _cleanup(st.session_state.tmp_paths)
            st.session_state.tmp_paths = []
            st.error(f"Pipeline error: {data}")

    except queue_module.Empty:
        pass  # Thread still running — rendering below will trigger rerun


# ---------------------------------------------------------------------------
# ② Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔬 ResearchMind MAS")
    st.caption("Multi-Agent Research Assistant")

    # ── Ollama status ──────────────────────────────────────────────────────
    ollama_ok = _ollama_running()
    if ollama_ok:
        st.success("Ollama is running", icon="✅")
    else:
        st.error("Ollama offline — run `ollama serve`", icon="🔴")

    st.divider()

    # ── Run mode ───────────────────────────────────────────────────────────
    run_mode = st.radio("Run Mode", ["Full Pipeline", "Single Agent"], horizontal=True)

    st.divider()

    # ── File uploads ───────────────────────────────────────────────────────
    st.markdown("**Upload Files**")

    proposal_file = st.file_uploader(
        "📄 Proposal PDF *(required)*",
        type=["pdf"],
        help="Primary document — analyzed as the base paper by all agents.",
    )
    paper_files = st.file_uploader(
        "📚 Literature PDFs *(optional)*",
        type=["pdf"],
        accept_multiple_files=True,
        help="Reference papers compared against the proposal.",
    )
    own_paper_file = st.file_uploader(
        "📝 Own Paper PDF *(optional)*",
        type=["pdf"],
        help="Agent 4 audits this. Defaults to the Proposal PDF if omitted.",
    )

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────
    st.markdown("**Settings**")

    # Single Agent selector
    selected_agent_label = ""
    if run_mode == "Single Agent":
        selected_agent_label = st.selectbox("Agent", SINGLE_AGENT_OPTIONS)
        node_key = AGENT_NODE_MAP[selected_agent_label]
        st.caption(AGENT_DESCRIPTIONS[node_key])

    model_name = st.selectbox("🤖 Model", MODELS)

    st.divider()

    # ── New Chat ───────────────────────────────────────────────────────────
    if st.session_state.messages or st.session_state.final_state:
        if st.button("✚ New Chat", use_container_width=True):
            if st.session_state.pipeline_stop:
                st.session_state.pipeline_stop.set()
            st.session_state.messages = []
            st.session_state.final_state = None
            st.session_state.single_result = None
            st.session_state.single_node = None
            st.session_state.pipeline_step = -1
            st.session_state.pipeline_running = False
            st.session_state.pipeline_topic = ""
            st.session_state.pipeline_state = None
            st.session_state.pipeline_result_q = None
            _cleanup(st.session_state.tmp_paths)
            st.session_state.tmp_paths = []
            st.rerun()
        st.divider()

    # ── Action buttons ─────────────────────────────────────────────────────
    pipeline_active = (
        st.session_state.pipeline_running
        or st.session_state.pipeline_step >= 0
    )

    if run_mode == "Full Pipeline":
        run_full = st.button(
            "▶ Run Full Pipeline",
            disabled=not ollama_ok or proposal_file is None or pipeline_active,
            use_container_width=True,
            type="primary",
        )
        if pipeline_active:
            if st.button("⏹ Stop Pipeline", use_container_width=True, type="secondary"):
                if st.session_state.pipeline_stop:
                    st.session_state.pipeline_stop.set()
                    st.session_state.pipeline_stopping = True
            step = st.session_state.pipeline_step
            if 0 <= step < len(AGENT_ORDER):
                st.info(
                    f"⏳ {AGENT_DISPLAY[AGENT_ORDER[step]]} is running…",
                    icon="🔄",
                )
        if proposal_file is None:
            st.caption("Upload a Proposal PDF to enable.")
    else:
        run_full = False
        run_single = st.button(
            f"▶ Run {selected_agent_label.split('·')[0].strip()}",
            disabled=not ollama_ok or proposal_file is None,
            use_container_width=True,
            type="primary",
        )
        if proposal_file is None:
            st.caption("Upload a Proposal PDF to enable.")

    # ── Download (post-run) ────────────────────────────────────────────────
    if run_mode == "Full Pipeline" and st.session_state.final_state:
        st.divider()
        st.download_button(
            "⬇ Download Full Report (PDF)",
            data=_state_to_pdf(st.session_state.final_state),
            file_name="researchmind_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# ③ Main area — Full Pipeline
# ---------------------------------------------------------------------------

if run_mode == "Full Pipeline":
    st.markdown("## ResearchMind MAS")
    st.caption(
        "Upload your Proposal PDF in the sidebar and click **▶ Run Full Pipeline**. "
        "The research topic is extracted automatically from the proposal."
    )

    # ── Handle Run button click ────────────────────────────────────────────
    if run_full:
        tmp_paths: List[str] = []
        proposal_path = _save_upload(proposal_file)
        tmp_paths.append(proposal_path)
        lit_paths = [_save_upload(f) for f in (paper_files or [])]
        tmp_paths.extend(lit_paths)
        own_path = _save_upload(own_paper_file) if own_paper_file else ""
        if own_path:
            tmp_paths.append(own_path)

        initial = _empty_state(
            proposal_pdf_path=proposal_path,
            paper_pdf_paths=lit_paths,
            own_paper_path=own_path,
            model_name=model_name,
        )

        # Reset pipeline state machine
        st.session_state.messages = []
        st.session_state.final_state = None
        st.session_state.pipeline_step = 0
        st.session_state.pipeline_state = dict(initial)
        st.session_state.pipeline_stop = threading.Event()
        st.session_state.pipeline_topic = "Analyzing proposal…"
        st.session_state.tmp_paths = tmp_paths

        # Show user bubble
        st.session_state.messages.append({
            "role": "user",
            "content": st.session_state.pipeline_topic,
        })

        _start_agent(0)
        st.rerun()

    # ── Render chat history ────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"### {AGENT_DISPLAY.get(msg['node'], msg['node'])}")
                AGENT_RENDERERS[msg["node"]](msg["state"])
                errs = msg["state"].get("errors", [])
                for e in errs:
                    st.error(e)

    # ── Generating indicator ───────────────────────────────────────────────
    if st.session_state.pipeline_running or 0 <= st.session_state.pipeline_step < len(AGENT_ORDER):
        step = st.session_state.pipeline_step
        if 0 <= step < len(AGENT_ORDER):
            node_name = AGENT_ORDER[step]
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**{AGENT_DISPLAY[node_name]}**")
                if st.session_state.pipeline_stopping:
                    st.markdown(
                        "<span style='color:#f4a261;font-size:0.9em'>"
                        "⏹ Stopping after this agent finishes…"
                        "</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='"
                        "display:inline-flex;gap:6px;align-items:center;margin-top:4px"
                        "'>"
                        "<span style='width:8px;height:8px;border-radius:50%;background:#6bb6ff;"
                        "animation:rmind 1.4s ease-in-out infinite;animation-delay:0s'></span>"
                        "<span style='width:8px;height:8px;border-radius:50%;background:#6bb6ff;"
                        "animation:rmind 1.4s ease-in-out infinite;animation-delay:0.2s'></span>"
                        "<span style='width:8px;height:8px;border-radius:50%;background:#6bb6ff;"
                        "animation:rmind 1.4s ease-in-out infinite;animation-delay:0.4s'></span>"
                        "<style>@keyframes rmind{"
                        "0%,80%,100%{opacity:.2;transform:scale(0.7)}"
                        "40%{opacity:1;transform:scale(1)}}"
                        "</style></span>",
                        unsafe_allow_html=True,
                    )
                time.sleep(0.4)
            st.rerun()


# ---------------------------------------------------------------------------
# ④ Main area — Single Agent
# ---------------------------------------------------------------------------

else:
    node_key = AGENT_NODE_MAP.get(selected_agent_label, "research_planner")

    st.markdown(f"## {selected_agent_label}")
    st.caption(AGENT_DESCRIPTIONS[node_key])

    prereqs = {
        "Agent 2 · Literature Review": "Agent 1",
        "Agent 3 · Gap & Hypothesis":  "Agent 1 and Agent 2",
    }
    if selected_agent_label in prereqs:
        st.info(
            f"**{prereqs[selected_agent_label]} will run silently** as prerequisites "
            "before this agent executes."
        )

    st.divider()

    # Show previous result
    if (
        st.session_state.single_result is not None
        and st.session_state.single_node == node_key
    ):
        result_state = st.session_state.single_result
        for err in result_state.get("errors", []):
            st.error(err)
        AGENT_RENDERERS[node_key](result_state)
        st.divider()
        st.download_button(
            "⬇ Download Agent Output (PDF)",
            data=_state_to_pdf(result_state),
            file_name=f"{node_key}_output.pdf",
            mime="application/pdf",
        )
    else:
        st.markdown("Upload files and click **▶ Run** in the sidebar to execute this agent.")

    # Handle Run button
    if run_single:
        tmp_paths = []
        try:
            proposal_path = _save_upload(proposal_file)
            tmp_paths.append(proposal_path)
            lit_paths = [_save_upload(f) for f in (paper_files or [])]
            tmp_paths.extend(lit_paths)
            own_path = _save_upload(own_paper_file) if own_paper_file else ""
            if own_path:
                tmp_paths.append(own_path)

            result = _run_single_agent(
                agent_label=selected_agent_label,
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
