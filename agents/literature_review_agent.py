from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from logger import log_agent_event
from state import ResearchMindState
from tools.paper_parser_tool import parse_multiple_papers

SECTION_LABELS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "literature_review": "literature_review",
    "methodology": "methodology",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "references": "references",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "using", "with",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _extract_terms(text: str) -> List[str]:
    tokens = []
    for raw_token in text.lower().replace(",", " ").replace(";", " ").split():
        token = "".join(ch for ch in raw_token if ch.isalnum() or ch == "-")
        if len(token) < 4 or token in STOPWORDS or token.isdigit():
            continue
        tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Paper summary builders
# ---------------------------------------------------------------------------

def _build_paper_summary(parsed_paper: Dict[str, Any]) -> Dict[str, Any]:
    sections = parsed_paper.get("sections", {})
    abstract = parsed_paper.get("abstract", "")
    return {
        "title": parsed_paper.get("title", "Untitled Paper"),
        "authors": parsed_paper.get("authors", ""),
        "abstract": abstract or "Abstract not found.",
        "introduction": sections.get(
            "introduction",
            "Introduction section not clearly available in the extracted text.",
        ),
        "methodology": sections.get(
            "methodology",
            "Methodology section not clearly available in the extracted text.",
        ),
        "findings": (
            sections.get("results", "")
            or sections.get("discussion", "")
            or "Findings section not clearly available in the extracted text."
        ),
        "key_findings": parsed_paper.get("key_findings", []),
        "citations": parsed_paper.get("references", []),
        "full_sections": sections,
    }


def _build_citation_map(parsed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    citation_map: Dict[str, Any] = {}
    for paper in parsed_papers:
        title = paper.get("title", "Untitled Paper")
        citation_map[title] = {
            "reference_count": len(paper.get("references", [])),
            "references": paper.get("references", []),
        }
    return citation_map


def _build_section_explanations(parsed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    explanations: Dict[str, Any] = {}
    for paper in parsed_papers:
        title = paper.get("title", "Untitled Paper")
        sections = paper.get("sections", {})
        paper_explanations = {}
        for section_name, label in SECTION_LABELS.items():
            content = sections.get(section_name, "").strip()
            paper_explanations[label] = (
                content if content else f"{label.title()} section not found in extracted text."
            )
        explanations[title] = paper_explanations
    return explanations


def _build_core_themes(
    research_topic: str,
    parsed_papers: List[Dict[str, Any]],
    limit: int = 5,
) -> List[str]:
    counter: Counter[str] = Counter()
    for paper in parsed_papers:
        combined_text = " ".join([
            paper.get("title", ""),
            paper.get("abstract", ""),
            " ".join(paper.get("key_findings", [])),
        ])
        counter.update(_extract_terms(combined_text))
    if research_topic:
        counter.update(_extract_terms(research_topic))
    return [term for term, _count in counter.most_common(limit)]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(model_name: str, prompt: str, num_predict: int = 1000) -> Dict[str, Any]:
    import ollama
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            format="json",
            stream=False,
            options={"temperature": 0.2, "num_predict": num_predict},
        )
        raw = response["response"].strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw, "error": "LLM did not return valid JSON"}
    except Exception as exc:
        return {"error": str(exc)}


def _analyze_primary_paper(
    primary_title: str,
    proposal: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    objectives = " ".join(proposal.get("objectives", [])) or ""
    scope = proposal.get("scope", "")
    methodology = proposal.get("methodology", "")
    raw_text = proposal.get("raw_text", "")
    abstract_hint = raw_text[:800] if raw_text else ""

    prompt = f"""You are an expert academic researcher. Analyze this research paper and extract structured information.

Title: {primary_title}

Abstract/Overview:
{abstract_hint}

Objectives:
{objectives[:600]}

Scope:
{scope[:400]}

Methodology:
{methodology[:600]}

Reply with valid JSON only, no markdown, no extra text:
{{
  "research_problem": "The main problem or gap this paper addresses",
  "research_objectives": "The specific objectives or research questions",
  "methodology_summary": "Summary of the research approach and methods",
  "key_contributions": "The unique contributions this paper makes to the field",
  "main_findings": "The main findings and results",
  "research_gaps_identified": "Research gaps identified or acknowledged in this paper",
  "limitations": "Limitations of this paper",
  "future_directions": "Suggested future research directions from this paper",
  "research_domain": "The specific research domain or field"
}}"""
    return _call_llm(model_name, prompt, num_predict=1200)


def _compare_paper_to_primary(
    primary_title: str,
    primary_analysis: Dict[str, Any],
    reference_paper: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    ref_title = reference_paper.get("title", "Unknown Paper")
    ref_abstract = reference_paper.get("abstract", "")
    ref_methodology = reference_paper.get("methodology", "")
    ref_findings = reference_paper.get("findings", "")

    primary_problem = primary_analysis.get("research_problem", "")
    primary_contributions = primary_analysis.get("key_contributions", "")
    primary_methodology = primary_analysis.get("methodology_summary", "")

    prompt = f"""You are an expert academic researcher performing a comparative literature review.

PRIMARY PAPER (the one being reviewed):
Title: {primary_title}
Research Problem: {primary_problem[:400]}
Methodology: {primary_methodology[:300]}
Key Contributions: {primary_contributions[:300]}

REFERENCE PAPER (for comparison):
Title: {ref_title}
Abstract: {ref_abstract[:600]}
Methodology: {ref_methodology[:400]}
Findings: {ref_findings[:400]}

Analyze how the reference paper relates to the primary paper. Reply with valid JSON only:
{{
  "relationship_type": "complementary/contradictory/foundational/extension/unrelated",
  "similarities": "Key similarities in domain, approach, or objectives",
  "differences": "Key differences in methodology, scope, or conclusions",
  "how_it_supports_primary": "How this reference paper provides context, evidence, or justification for the primary paper",
  "gaps_it_highlights": "Research gaps or limitations in this reference paper that the primary paper may address",
  "relevance": "high/medium/low"
}}"""
    return _call_llm(model_name, prompt, num_predict=800)


def _generate_synthesis(
    primary_title: str,
    primary_analysis: Dict[str, Any],
    paper_comparisons: List[Dict[str, Any]],
    research_topic: str,
    model_name: str,
) -> Dict[str, Any]:
    primary_problem = primary_analysis.get("research_problem", "")
    primary_contributions = primary_analysis.get("key_contributions", "")
    primary_gaps = primary_analysis.get("research_gaps_identified", "")

    comparisons_text = ""
    for idx, item in enumerate(paper_comparisons[:6], 1):
        title = item.get("paper_title", f"Paper {idx}")
        comp = item.get("comparison", {})
        rel = comp.get("relationship_type", "")
        support = comp.get("how_it_supports_primary", "")
        gaps = comp.get("gaps_it_highlights", "")
        comparisons_text += (
            f"\n[{idx}] {title} ({rel}): {support[:200]} Gaps: {gaps[:150]}"
        )

    topic_line = f"Research Topic: {research_topic}\n" if research_topic else ""

    prompt = f"""You are an expert academic researcher writing a synthesis for a literature review.

{topic_line}PRIMARY PAPER: {primary_title}
Research Problem: {primary_problem[:400]}
Key Contributions: {primary_contributions[:300]}
Identified Gaps: {primary_gaps[:300]}

RELATED WORKS SUMMARY:{comparisons_text}

Write a comprehensive literature review synthesis. Reply with valid JSON only:
{{
  "narrative": "A 3-5 sentence academic narrative reviewing the literature and positioning the primary paper among related works",
  "research_positioning": "How the primary paper is positioned relative to existing work in the field",
  "gaps_addressed": "Research gaps identified across the literature that the primary paper addresses",
  "methodological_comparison": "How the primary paper's methodology compares to approaches in related work",
  "novelty_assessment": "Assessment of the primary paper's unique contributions not found in related work",
  "future_research_directions": "Directions for future research based on gaps across the literature"
}}"""
    return _call_llm(model_name, prompt, num_predict=1200)


def _generate_literature_review_report(
    state: ResearchMindState,
    paper_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    model_name = state.get("model_name", "qwen2.5:1.5b")
    research_topic = state.get("research_topic", "")
    proposal = state.get("proposal_extracted", {})
    primary_title = proposal.get("title", research_topic) or "Research Proposal"

    print(f"  [LLM] Analyzing primary paper: {primary_title}")
    primary_analysis = _analyze_primary_paper(primary_title, proposal, model_name)

    paper_comparisons = []
    for paper in paper_summaries:
        print(f"  [LLM] Comparing: {paper.get('title', 'Unknown')}")
        comparison = _compare_paper_to_primary(
            primary_title, primary_analysis, paper, model_name
        )
        paper_comparisons.append({
            "paper_title": paper.get("title", "Unknown"),
            "comparison": comparison,
        })

    print("  [LLM] Generating synthesis...")
    synthesis = _generate_synthesis(
        primary_title, primary_analysis, paper_comparisons, research_topic, model_name
    )

    return {
        "primary_paper_title": primary_title,
        "primary_paper_analysis": primary_analysis,
        "primary_paper_citations": proposal.get("references", []),
        "comparison_count": len(paper_comparisons),
        "paper_comparisons": paper_comparisons,
        "synthesis": synthesis,
    }


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def literature_review_agent(state: ResearchMindState) -> ResearchMindState:
    agent_name = "literature_review_agent"
    paper_paths = state.get("paper_pdf_paths", [])

    state = log_agent_event(
        state,
        agent_name=agent_name,
        event_type="START",
        input_data={
            "research_topic": state.get("research_topic", ""),
            "paper_count": len(paper_paths),
        },
    )

    if not paper_paths:
        message = "No paper paths were provided to the literature review agent."
        state = log_agent_event(
            state,
            agent_name=agent_name,
            event_type="ERROR",
            output_data={"message": message},
        )
        errors = list(state.get("errors", []))
        errors.append(message)
        return {**state, "errors": errors}

    state = log_agent_event(
        state,
        agent_name=agent_name,
        event_type="TOOL_CALL",
        input_data={"tool": "parse_multiple_papers", "paper_count": len(paper_paths)},
    )

    try:
        parsed_papers = parse_multiple_papers(paper_paths)
    except Exception as exc:
        message = f"Failed to parse papers: {exc}"
        state = log_agent_event(
            state,
            agent_name=agent_name,
            event_type="ERROR",
            output_data={"message": message},
        )
        errors = list(state.get("errors", []))
        errors.append(message)
        return {**state, "errors": errors}

    paper_summaries = [_build_paper_summary(paper) for paper in parsed_papers]
    citation_map = _build_citation_map(parsed_papers)
    section_explanations = _build_section_explanations(parsed_papers)
    core_themes = _build_core_themes(state.get("research_topic", ""), parsed_papers)

    state = log_agent_event(
        state,
        agent_name=agent_name,
        event_type="OUTPUT",
        output_data={
            "parsed_count": len(parsed_papers),
            "paper_summaries": len(paper_summaries),
            "citation_map_keys": len(citation_map),
            "core_theme_count": len(core_themes),
        },
    )

    # LLM-generated report: primary paper analysis + per-paper comparisons + synthesis
    print("\nLiterature Review Agent: generating LLM report...")
    try:
        literature_review_report = _generate_literature_review_report(
            {**state, "paper_pdf_paths": paper_paths},
            paper_summaries,
        )
    except Exception as exc:
        literature_review_report = {"error": f"LLM report generation failed: {exc}"}

    state = log_agent_event(
        state,
        agent_name=agent_name,
        event_type="OUTPUT",
        output_data={"status": "completed", "llm_report_generated": "error" not in literature_review_report},
    )

    return {
        **state,
        "paper_summaries": paper_summaries,
        "citation_map": citation_map,
        "section_explanations": section_explanations,
        "core_themes": core_themes,
        "literature_review_report": literature_review_report,
    }
