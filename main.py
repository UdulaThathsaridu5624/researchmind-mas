from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ollama

from graph import build_graph
from state import ResearchMindState


def _check_ollama() -> None:
    """Verify Ollama is running. Exit with a clear message if not."""
    try:
        ollama.list()
    except Exception:
        print(
            "ERROR: Cannot reach Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            "Then pull the model if you haven't already:\n"
            "  ollama pull gemma4:e2b"
        )
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ResearchMind MAS - Multi-Agent Academic Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py \
    --research-topic "Federated Learning for Healthcare" \
    --proposal-pdf ./my_proposal.pdf

  python main.py \
    --research-topic "NLP in Education" \
    --proposal-pdf ./proposal.pdf \
    --paper-pdfs ./paper1.pdf ./paper2.pdf \
    --own-paper ./my_paper.pdf \
    --output ./outputs/report.json
        """,
    )
    parser.add_argument(
        "--research-topic",
        required=True,
        help="The research topic or title of the study.",
    )
    parser.add_argument(
        "--proposal-pdf",
        required=True,
        help="Path to the research proposal PDF.",
    )
    parser.add_argument(
        "--paper-pdfs",
        nargs="*",
        default=[],
        metavar="PDF",
        help="Paths to literature review PDFs (0 or more).",
    )
    parser.add_argument(
        "--own-paper",
        default="",
        help="Path to your own paper PDF for auditing (optional).",
    )
    parser.add_argument(
        "--output",
        default="outputs/final_report.json",
        help="Path to write the final JSON state report (default: outputs/final_report.json).",
    )
    parser.add_argument(
        "--model",
        default="gemma4:e2b",
        help="Ollama model name to use (default: gemma4:e2b).",
    )
    return parser.parse_args()


def _build_initial_state(args: argparse.Namespace) -> ResearchMindState:
    """Construct the fully initialized starting state for the graph."""
    return ResearchMindState(
        research_topic=args.research_topic,
        proposal_pdf_path=args.proposal_pdf,
        paper_pdf_paths=args.paper_pdfs,
        own_paper_path=args.own_paper,
        model_name=args.model,
        # Agent outputs - all empty at start
        proposal_extracted={},
        implementation_plan="",
        timeline={},
        suggested_resources=[],
        paper_summaries=[],
        citation_map={},
        core_themes=[],
        section_explanations={},
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
        final_report_path=args.output,
        agent_logs=[],
        errors=[],
    )


def main() -> None:
    args = _parse_args()
    _check_ollama()

    print("\nResearchMind MAS starting...")
    print(f"  Topic   : {args.research_topic}")
    print(f"  Proposal: {args.proposal_pdf}")
    print(f"  Model   : {args.model}\n")

    initial_state = _build_initial_state(args)
    graph = build_graph()

    print("Running agent pipeline...\n")
    final_state: ResearchMindState = graph.invoke(initial_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(final_state), f, indent=2, default=str)

    print(f"\nDone. Report written to: {output_path}")
    print(f"Agent log entries     : {len(final_state.get('agent_logs', []))}")

    if final_state.get("errors"):
        print("\nErrors encountered during run:")
        for err in final_state["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
