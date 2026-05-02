# ResearchMind MAS

A locally-hosted Multi-Agent System (MAS) that automates academic research assistance.  
Built for SLIIT SE4010 — Current Trends in Software Engineering, Assignment 2.

---

## Overview

ResearchMind MAS orchestrates four specialised agents through a [LangGraph](https://github.com/langchain-ai/langgraph) pipeline.  
Each agent reads from and writes to a shared `ResearchMindState` TypedDict — no message passing, no network calls, everything runs on your machine.

| Agent | Member | Responsibility |
|-------|--------|---------------|
| **Agent 1 — Research Planner** | Udula | Extracts objectives, scope, methodology from proposal PDF; generates an implementation plan and timeline via Ollama LLM |
| **Agent 2 — Literature Review** | Tharindu | Parses reference PDFs, compares each against the proposal, produces LLM-written synthesis and gap narrative |
| **Agent 3 — Gap & Hypothesis** | Madhini | Detects research gaps and contradictions across papers; generates hypotheses and a positioning statement (deterministic, no LLM) |
| **Agent 4 — Paper Auditor** | Jameela | Audits the paper for missing sections, TF-IDF plagiarism score, and citation inconsistencies; writes detailed LLM feedback |

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Default model: `gemma4:e2b` (pull once with `ollama pull gemma4:e2b`)

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd researchmind-mas

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start Ollama (separate terminal)
ollama serve

# 4. Pull the default model
ollama pull gemma4:e2b
```

---

## Usage

### CLI (recommended)

```bash
python main.py \
  --research-topic "Federated Learning for Healthcare" \
  --proposal-pdf ./data/papers/proposal.pdf

# With literature PDFs and own paper for auditing
python main.py \
  --research-topic "NLP in Education" \
  --proposal-pdf ./proposal.pdf \
  --paper-pdfs ./paper1.pdf ./paper2.pdf ./paper3.pdf \
  --own-paper ./my_paper.pdf \
  --output ./outputs/report.json \
  --model gemma4:e2b
```

Output is written to `outputs/reports/final_report.json`.  
Agent logs are written to `logs/session_<timestamp>.jsonl`.

### Streamlit UI

```bash
python -m streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

- **Full Pipeline** — upload your Proposal PDF, click ▶ Run Full Pipeline; research topic is auto-extracted from the document
- **Single Agent** — run one agent independently with prerequisites silently pre-run

---

## Running Tests

```bash
# Unit tests (no Ollama required)
pytest tests/ -m "not integration and not llm_judge" -q

# Integration test (Ollama must be running)
pytest tests/ -m integration -v
```

Expected: **54 unit tests pass**.

---

## Project Structure

```
researchmind-mas/
├── agents/
│   ├── research_planner_agent.py   # Agent 1 — proposal analysis + planning
│   ├── literature_review_agent.py  # Agent 2 — literature comparison
│   ├── gap_hypothesis_agent.py     # Agent 3 — gap detection + hypotheses
│   └── paper_auditor_agent.py      # Agent 4 — audit + feedback
├── tools/
│   ├── proposal_analyzer_tool.py   # PDF extraction for proposals/reports
│   ├── paper_parser_tool.py        # PDF extraction for literature papers
│   ├── gap_analyzer_tool.py        # Gap detection, contradiction scoring
│   └── paper_audit_tool.py         # Section, plagiarism, citation checks
├── tests/
│   ├── test_planner_agent.py       # 9 unit tests — Agent 1
│   ├── test_literature_agent.py    # 11 unit tests — Agent 2
│   ├── test_gap_agent.py           # 12 unit tests — Agent 3 tool
│   ├── test_gap_hypothesis_agent.py# 6 unit tests — Agent 3 agent
│   └── test_auditor_agent.py       # 11 unit tests — Agent 4
├── state.py          # Shared ResearchMindState TypedDict
├── graph.py          # LangGraph pipeline — 4 nodes + error edges
├── logger.py         # Structured JSONL observability logger
├── main.py           # CLI entry point
├── app.py            # Streamlit UI
├── requirements.txt
└── pytest.ini
```

---

## Architecture

```
Proposal PDF
    │
    ▼
[Agent 1: Research Planner]
  proposal_extracted, implementation_plan, timeline, suggested_resources
    │
    ▼
[Agent 2: Literature Review]  ◄── Literature PDFs
  paper_summaries, citation_map, core_themes, literature_review_report
    │
    ▼
[Agent 3: Gap & Hypothesis]
  identified_gaps, gap_frequency_scores, hypotheses, positioning_statement, contradiction_pairs
    │
    ▼
[Agent 4: Paper Auditor]  ◄── Own Paper PDF (optional, defaults to Proposal)
  formatting_issues, missing_sections, plagiarism_score, integrity_score, audit_feedback
    │
    ▼
outputs/reports/final_report.json
outputs/audit_report.md
logs/session_<timestamp>.jsonl
```

All agents share a single `ResearchMindState` dict. Each agent reads its predecessors' outputs and appends its own. The LangGraph `StateGraph` routes errors to a terminal node rather than halting silently.

---

## Observability

Every agent logs structured events (START, TOOL_CALL, OUTPUT, ERROR) to `logs/session_<timestamp>.jsonl` via `logger.py`. Each entry includes agent name, event type, timestamp, and input/output data — suitable for debugging or post-hoc analysis.

---

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4:e2b` | Any Ollama model name |
| `--output` | `outputs/reports/final_report.json` | JSON state dump path |
| `--own-paper` | *(empty)* | Agent 4 falls back to Proposal PDF if omitted |
