from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_ollama import OllamaLLM

from logger import log_agent_event
from state import ResearchMindState
from tools.paper_audit_tool import paper_audit_tool

# ====================================================================== #
# Agent persona & system prompt
# ====================================================================== #

SYSTEM_PROMPT = """You are a rigorous academic paper reviewer with expertise in \
evaluating research papers for quality, completeness, and academic integrity.

You will be given the results of an automated audit of a student's paper. \
Your job is to turn these raw findings into clear, specific, and actionable \
written feedback that the student can use to improve their paper.

Rules:
- Be constructive and specific — do NOT say "improve your paper". Say exactly what is wrong and how to fix it.
- Address every issue found in the audit data.
- If the plagiarism score is above 0.5, flag it clearly and suggest how to rephrase or cite properly.
- If sections are missing, explain what each section should contain.
- If citations are inconsistent, explain the correct citation format.
- End with a short "Strengths" paragraph noting what appears to be done well.
- Write in plain academic English. No bullet points — use proper paragraphs.
- Keep your response between 200 and 400 words."""


# ====================================================================== #
# Agent entry point
# ====================================================================== #

def paper_auditor_agent(state: ResearchMindState) -> ResearchMindState:
    """
    LangGraph node — Paper Auditor Agent (Member 4: Jameela).

    Reads the student's own paper PDF and reference paper summaries from state,
    runs paper_audit_tool to perform three automated checks (section detection,
    TF-IDF plagiarism scoring, citation cross-referencing), then calls the local
    Ollama LLM to generate human-readable audit feedback.

    Finally writes a formatted Markdown report to /outputs/audit_report.md.

    Writes to state:
        formatting_issues  (List[str])
        missing_sections   (List[str])
        plagiarism_score   (float)
        integrity_score    (float)
        audit_feedback     (str)
        final_report_path  (str)

    Args:
        state: Current ResearchMindState passed by LangGraph.

    Returns:
        Updated ResearchMindState.
    """
    model_name: str = state.get("model_name", "llama3.2:1b")
    own_paper_path: str = state.get("own_paper_path", "")

    # Log agent start
    state = log_agent_event(
        state,
        agent_name="paper_auditor_agent",
        event_type="START",
        input_data={
            "own_paper_path": own_paper_path,
            "model_name": model_name,
            "reference_papers_available": len(state.get("paper_summaries", [])),
        },
    )

    # Guard: if no own paper, fall back to auditing the proposal itself
    if not own_paper_path:
        proposal_path: str = state.get("proposal_pdf_path", "")
        if proposal_path:
            own_paper_path = proposal_path
            state = log_agent_event(
                state,
                agent_name="paper_auditor_agent",
                event_type="OUTPUT",
                output_data={"info": "No own_paper_path — auditing proposal PDF instead."},
            )
        else:
            msg = "paper_auditor_agent: no own_paper_path or proposal_pdf_path provided — skipping audit."
            state = {
                **state,
                "formatting_issues": [],
                "missing_sections": [],
                "plagiarism_score": 0.0,
                "integrity_score": 0.0,
                "audit_feedback": "No paper was provided for auditing.",
                "final_report_path": "",
            }
            state = log_agent_event(
                state,
                agent_name="paper_auditor_agent",
                event_type="OUTPUT",
                output_data={"skipped": True, "reason": msg},
            )
            return state

    # ------------------------------------------------------------------ #
    # Step 1 — Run the audit tool
    # ------------------------------------------------------------------ #
    paper_summaries: List[Dict[str, Any]] = state.get("paper_summaries", [])
    # Pull Agent 3's outputs so the LLM feedback can evaluate gap alignment
    identified_gaps: List[str] = state.get("identified_gaps", [])
    hypotheses: List[str] = state.get("hypotheses", [])
    positioning_statement: str = state.get("positioning_statement", "")

    try:
        audit_result: Dict[str, Any] = paper_audit_tool(
            own_paper_path=own_paper_path,
            paper_summaries=paper_summaries,
        )
    except (FileNotFoundError, ValueError) as exc:
        error_msg = f"paper_auditor_agent: paper_audit_tool failed — {exc}"
        state = {
            **state,
            "errors": list(state.get("errors", [])) + [error_msg],
        }
        state = log_agent_event(
            state,
            agent_name="paper_auditor_agent",
            event_type="ERROR",
            output_data={"error": error_msg},
        )
        return state

    # Log the tool call
    state = log_agent_event(
        state,
        agent_name="paper_auditor_agent",
        event_type="TOOL_CALL",
        tool_calls=[
            {
                "tool": "paper_audit_tool",
                "input": {"own_paper_path": own_paper_path},
                "output": {
                    "missing_sections":  audit_result["missing_sections"],
                    "plagiarism_score":  audit_result["plagiarism_score"],
                    "integrity_score":   audit_result["integrity_score"],
                    "citation_issues_count": len(audit_result["citation_issues"]),
                },
            }
        ],
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Build the LLM prompt from audit findings + Agent 3 outputs
    # ------------------------------------------------------------------ #
    user_prompt = _build_llm_prompt(
        audit_result, own_paper_path, identified_gaps, hypotheses, positioning_statement
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    # ------------------------------------------------------------------ #
    # Step 3 — Call Ollama LLM to generate readable feedback
    # ------------------------------------------------------------------ #
    try:
        llm = OllamaLLM(model=model_name, temperature=0.3)
        audit_feedback: str = llm.invoke(full_prompt).strip()
    except Exception as exc:
        # If LLM fails, fall back to a structured text summary
        audit_feedback = _build_fallback_feedback(audit_result)
        error_msg = f"paper_auditor_agent: Ollama call failed — {exc}. Using fallback feedback."
        state = {**state, "errors": list(state.get("errors", [])) + [error_msg]}

    # ------------------------------------------------------------------ #
    # Step 4 — Write Markdown report to /outputs/
    # ------------------------------------------------------------------ #
    report_path: str = _write_markdown_report(
        audit_result=audit_result,
        audit_feedback=audit_feedback,
        own_paper_path=own_paper_path,
        research_topic=state.get("research_topic", ""),
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Write all results back to state
    # ------------------------------------------------------------------ #
    state = {
        **state,
        "formatting_issues": audit_result["formatting_issues"],
        "missing_sections":  audit_result["missing_sections"],
        "plagiarism_score":  audit_result["plagiarism_score"],
        "integrity_score":   audit_result["integrity_score"],
        "audit_feedback":    audit_feedback,
        "final_report_path": report_path,
    }

    state = log_agent_event(
        state,
        agent_name="paper_auditor_agent",
        event_type="OUTPUT",
        output_data={
            "missing_sections_count":  len(audit_result["missing_sections"]),
            "formatting_issues_count": len(audit_result["formatting_issues"]),
            "plagiarism_score":        audit_result["plagiarism_score"],
            "integrity_score":         audit_result["integrity_score"],
            "audit_feedback_length":   len(audit_feedback),
            "report_written_to":       report_path,
        },
    )

    return state


# ====================================================================== #
# Private helpers
# ====================================================================== #

def _build_llm_prompt(
    audit_result: Dict[str, Any],
    paper_path: str,
    identified_gaps: List[str],
    hypotheses: List[str],
    positioning_statement: str,
) -> str:
    """
    Build a detailed prompt for the LLM from the tool's structured audit result and
    the research gap analysis produced by Agent 3.

    Args:
        audit_result:          Dict returned by paper_audit_tool.
        paper_path:            Path to the paper (used for context).
        identified_gaps:       Gaps identified by gap_hypothesis_agent (Agent 3).
        hypotheses:            Hypotheses generated by gap_hypothesis_agent (Agent 3).
        positioning_statement: Positioning statement from gap_hypothesis_agent (Agent 3).

    Returns:
        Formatted prompt string.
    """
    missing = audit_result["missing_sections"]
    formatting = audit_result["formatting_issues"]
    plagiarism = audit_result["plagiarism_score"]
    citations = audit_result["citation_issues"]
    integrity = audit_result["integrity_score"]

    missing_str = ", ".join(missing) if missing else "None — all required sections found."
    formatting_str = "\n".join(f"  - {i}" for i in formatting) if formatting else "  - None detected."
    citation_str = "\n".join(f"  - {c}" for c in citations) if citations else "  - No issues detected."
    gaps_str = "\n".join(f"  - {g}" for g in identified_gaps) if identified_gaps else "  - Not yet identified."
    hyp_str = "\n".join(f"  - {h}" for h in hypotheses[:3]) if hypotheses else "  - Not yet generated."
    position_str = positioning_statement if positioning_statement else "Not available."

    prompt = f"""You are reviewing a paper submitted at: {paper_path}

AUTOMATED AUDIT RESULTS:

1. Missing sections: {missing_str}

2. Formatting issues:
{formatting_str}

3. Plagiarism / similarity score: {plagiarism:.2f} out of 1.0
   (0.0 = fully original, 1.0 = identical to a reference paper)

4. Citation consistency issues:
{citation_str}

5. Overall integrity score: {integrity:.2f} out of 1.0

RESEARCH GAP ANALYSIS (from literature review):

6. Research gaps identified from related literature:
{gaps_str}

7. Research hypotheses the paper should address:
{hyp_str}

8. Suggested research positioning:
  {position_str}

Based on these findings, write comprehensive, actionable feedback for the student. \
Include an evaluation of whether the paper addresses the identified research gaps and \
whether its methodology aligns with the proposed hypotheses."""

    return prompt


def _build_fallback_feedback(audit_result: Dict[str, Any]) -> str:
    """
    Generate plain-text feedback without the LLM, used when Ollama is unavailable.

    Args:
        audit_result: Dict returned by paper_audit_tool.

    Returns:
        Plain-text feedback string.
    """
    lines = ["AUTOMATED AUDIT FEEDBACK (LLM unavailable)\n"]

    missing = audit_result["missing_sections"]
    if missing:
        lines.append(f"Missing sections: {', '.join(missing)}.")
        lines.append("Each missing section must be added before submission.")

    if audit_result["plagiarism_score"] > 0.5:
        lines.append(
            f"High similarity score ({audit_result['plagiarism_score']:.2f}) detected. "
            "Please ensure all content is original and properly cited."
        )

    for issue in audit_result["citation_issues"]:
        lines.append(f"Citation issue: {issue}")

    for issue in audit_result["formatting_issues"]:
        lines.append(f"Formatting: {issue}")

    lines.append(f"\nOverall integrity score: {audit_result['integrity_score']:.2f} / 1.0")
    return "\n".join(lines)


def _write_markdown_report(
    audit_result: Dict[str, Any],
    audit_feedback: str,
    own_paper_path: str,
    research_topic: str,
) -> str:
    """
    Write a formatted Markdown audit report to the /outputs/ directory.

    Args:
        audit_result:    Dict returned by paper_audit_tool.
        audit_feedback:  LLM-generated feedback string.
        own_paper_path:  Path to the student's paper.
        research_topic:  Research topic from state.

    Returns:
        Path string where the report was written.
    """
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "audit_report.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    missing = audit_result["missing_sections"]
    formatting = audit_result["formatting_issues"]
    citations = audit_result["citation_issues"]
    plagiarism = audit_result["plagiarism_score"]
    integrity = audit_result["integrity_score"]

    # Score badge
    if integrity >= 0.8:
        badge = "GOOD"
    elif integrity >= 0.6:
        badge = "NEEDS IMPROVEMENT"
    else:
        badge = "POOR — MAJOR REVISIONS REQUIRED"

    lines = [
        "# Paper Audit Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Paper:** `{own_paper_path}`  ",
        f"**Research topic:** {research_topic or 'N/A'}  ",
        f"**Overall integrity score:** {integrity:.2f} / 1.00 — {badge}",
        "",
        "---",
        "",
        "## 1. Audit Summary",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| Missing sections | {len(missing)} |",
        f"| Formatting issues | {len(formatting)} |",
        f"| Plagiarism / similarity score | {plagiarism:.2f} |",
        f"| Citation inconsistencies | {len(citations)} |",
        f"| **Integrity score** | **{integrity:.2f} / 1.00** |",
        "",
        "---",
        "",
        "## 2. Missing Sections",
        "",
    ]

    if missing:
        for section in missing:
            lines.append(f"- **{section}** — This section was not found in your paper.")
    else:
        lines.append("All required sections (Abstract, Introduction, Methodology, Conclusion, References) are present.")

    lines += [
        "",
        "---",
        "",
        "## 3. Formatting Issues",
        "",
    ]

    if formatting:
        for issue in formatting:
            lines.append(f"- {issue}")
    else:
        lines.append("No formatting issues detected.")

    lines += [
        "",
        "---",
        "",
        "## 4. Plagiarism Analysis",
        "",
        f"**Similarity score: {plagiarism:.2f}**",
        "",
    ]

    if plagiarism > 0.7:
        lines.append(
            "> WARNING: High similarity detected with reference papers. "
            "Significant portions of your paper closely match existing work. "
            "Review your writing and ensure all borrowed ideas are properly cited and paraphrased."
        )
    elif plagiarism > 0.4:
        lines.append(
            "> CAUTION: Moderate similarity detected. Some sections may overlap with "
            "existing literature. Verify that all referenced content is properly cited."
        )
    else:
        lines.append(
            "> The paper appears to be largely original based on the similarity analysis."
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Citation Consistency Check",
        "",
    ]

    if citations:
        for issue in citations:
            lines.append(f"- {issue}")
    else:
        lines.append("All detected citations appear to have matching reference entries.")

    lines += [
        "",
        "---",
        "",
        "## 6. Detailed Feedback",
        "",
        audit_feedback,
        "",
        "---",
        "",
        "## 7. Recommendations",
        "",
        "Based on the audit above, the following actions are recommended before submission:",
        "",
    ]

    if missing:
        lines.append(f"1. Add the missing section(s): {', '.join(missing)}.")
    if plagiarism > 0.4:
        lines.append(f"{'2' if missing else '1'}. Revise sections with high similarity to existing papers.")
    if citations:
        lines.append(f"- Resolve all {len(citations)} citation inconsistencies identified above.")
    if formatting:
        lines.append(f"- Address formatting issues: {'; '.join(formatting)}")

    lines += [
        "",
        "---",
        "",
        "_This report was generated automatically by the ResearchMind Paper Auditor Agent._",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)