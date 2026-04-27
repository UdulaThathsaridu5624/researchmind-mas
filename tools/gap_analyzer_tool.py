from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


_ASPECTS: Dict[str, List[str]] = {
	"accuracy": ["accuracy", "f1", "precision", "recall", "auc", "performance"],
	"efficiency": ["latency", "throughput", "runtime", "compute", "efficiency"],
	"privacy": ["privacy", "confidentiality", "anonym", "dp", "differential privacy"],
	"robustness": ["robust", "noise", "adversarial", "out-of-distribution", "ood"],
	"fairness": ["fair", "bias", "equity", "demographic"],
	"generalization": ["generaliz", "cross-domain", "transfer", "external validity"],
}

_POSITIVE_PAT = re.compile(r"\b(improv(?:e|ed|es|ement)|outperform(?:s|ed)?|better|increase(?:d)?|gain(?:ed)?)\b")
_NEGATIVE_PAT = re.compile(
	r"\b(no\s+improv(?:e|ement)|worse|decrease(?:d)?|drop(?:ped)?|trade[- ]?off|degrad(?:e|ed|ation)|limited)\b"
)

_GAP_PATTERNS: List[Tuple[str, str]] = [
	("generalization", r"\b(generaliz|cross[- ]domain|external\s+validity|domain\s+shift)\b"),
	("scalability", r"\b(scale|scalab|large[- ]scale|many\s+nodes|high\s+load)\b"),
	("privacy_security", r"\b(privacy|security|attack|leak|confidential)\b"),
	("interpretability", r"\b(interpret|explain|black\s*box|transparen)\b"),
	("robustness", r"\b(robust|adversarial|noise|ood|out[- ]of[- ]distribution)\b"),
	("fairness", r"\b(fair|bias|equity|demograph)\b"),
	("real_world_validation", r"\b(real[- ]world|deployment|clinical|production|field\s+study)\b"),
	("resource_constraints", r"\b(low[- ]resource|edge|memory|energy|compute\s+cost)\b"),
	("long_term_evaluation", r"\b(long[- ]term|longitudinal|follow[- ]up|drift\s+over\s+time)\b"),
]


def gap_analyzer_tool(
	paper_summaries: List[Dict[str, Any]],
	citation_map: Dict[str, Any],
	core_themes: List[str],
	research_topic: str,
) -> Dict[str, Any]:
	"""
	Identify research gaps, hypotheses, and contradiction pairs from literature artifacts.

	This tool is deterministic and does not require an LLM, so it can run in unit tests.
	"""
	papers = paper_summaries or []
	themes = [t.strip() for t in (core_themes or []) if isinstance(t, str) and t.strip()]

	corpus_parts: List[str] = []
	for paper in papers:
		corpus_parts.append(_paper_to_text(paper))
	corpus_parts.extend(themes)
	corpus_text = "\n".join(corpus_parts).lower()

	gap_scores = _compute_gap_scores(corpus_text, citation_map, themes)
	identified_gaps = _build_gap_statements(gap_scores, research_topic)
	contradiction_pairs = _find_contradictions(papers)
	hypotheses = _build_hypotheses(identified_gaps, themes, research_topic)
	positioning_statement = _build_positioning_statement(identified_gaps, themes, research_topic)

	return {
		"identified_gaps": identified_gaps,
		"gap_frequency_scores": gap_scores,
		"hypotheses": hypotheses,
		"positioning_statement": positioning_statement,
		"contradiction_pairs": contradiction_pairs,
	}


def _paper_to_text(paper: Dict[str, Any]) -> str:
	chunks: List[str] = []
	for key, value in paper.items():
		if isinstance(value, str):
			chunks.append(value)
		elif isinstance(value, list):
			chunks.extend([str(v) for v in value if isinstance(v, (str, int, float))])
		elif isinstance(value, dict):
			chunks.extend([str(v) for v in value.values() if isinstance(v, (str, int, float))])
	return "\n".join(chunks)


def _compute_gap_scores(
	corpus_text: str,
	citation_map: Dict[str, Any],
	core_themes: List[str],
) -> Dict[str, float]:
	counts: Counter[str] = Counter()

	limitation_sentences = [
		s
		for s in re.split(r"[.!?]\s+", corpus_text)
		if re.search(r"\b(limit|underexplor|few|lack|open\s+question|future\s+work|challenge)\b", s)
	]
	limitation_bonus = max(1, len(limitation_sentences))

	for gap_key, pattern in _GAP_PATTERNS:
		counts[gap_key] += len(re.findall(pattern, corpus_text))
		if counts[gap_key] > 0:
			counts[gap_key] += limitation_bonus

	# Theme coverage adjustment: low-citation themes usually indicate potential gaps.
	for theme in core_themes:
		theme_l = theme.lower()
		if theme_l and theme_l in corpus_text:
			counts["theme_coverage"] += 1

	if isinstance(citation_map, dict):
		for key, value in citation_map.items():
			low_signal = _is_low_citation(value)
			if low_signal:
				counts["theme_coverage"] += 2
			if isinstance(key, str) and re.search(r"contradict|conflict|inconsisten", key, re.IGNORECASE):
				counts["generalization"] += 1

	if not counts:
		counts["generalization"] = 1
		counts["real_world_validation"] = 1

	max_count = max(counts.values()) if counts else 1
	if max_count <= 0:
		max_count = 1
	return {k: round(v / max_count, 3) for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))}


def _is_low_citation(value: Any) -> bool:
	if isinstance(value, int):
		return value <= 1
	if isinstance(value, float):
		return value <= 1.0
	if isinstance(value, list):
		return len(value) <= 1
	if isinstance(value, dict):
		citations = value.get("citations")
		if isinstance(citations, list):
			return len(citations) <= 1
		score = value.get("count")
		if isinstance(score, (int, float)):
			return score <= 1
	return False


def _build_gap_statements(gap_scores: Dict[str, float], topic: str) -> List[str]:
	top_keys = list(gap_scores.keys())[:5] if gap_scores else []
	pretty = {
		"generalization": "limited cross-domain generalization",
		"scalability": "insufficient large-scale validation",
		"privacy_security": "incomplete privacy and security analysis",
		"interpretability": "weak model interpretability evidence",
		"robustness": "insufficient robustness under noisy/adversarial settings",
		"fairness": "limited fairness and bias evaluation",
		"real_world_validation": "lack of real-world deployment studies",
		"resource_constraints": "limited evaluation under low-resource constraints",
		"long_term_evaluation": "missing longitudinal performance tracking",
		"theme_coverage": "uneven coverage across reported themes",
	}
	gaps: List[str] = []
	for key in top_keys:
		phr = pretty.get(key, key.replace("_", " "))
		gaps.append(f"In {topic}, current literature shows {phr}.")

	if not gaps:
		gaps = [
			f"In {topic}, literature indicates underexplored problem settings and limited validation depth.",
			f"In {topic}, evidence across studies remains fragmented and difficult to reproduce.",
		]
	return gaps


def _find_contradictions(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	claims_by_aspect: Dict[str, List[Dict[str, str]]] = defaultdict(list)
	for idx, paper in enumerate(papers):
		title = str(paper.get("title") or paper.get("paper_title") or f"paper_{idx + 1}")
		text = _paper_to_text(paper)
		if not text.strip():
			continue
		sentences = re.split(r"[.!?]\s+", text)

		for sentence in sentences:
			clauses = re.split(r"\bbut\b|\bhowever\b|;", sentence, flags=re.IGNORECASE)
			for clause in clauses:
				sent_l = clause.lower().strip()
				if not sent_l:
					continue
				polarity = _claim_polarity(sent_l)
				if polarity == "neutral":
					continue
				for aspect, words in _ASPECTS.items():
					if any(word in sent_l for word in words):
						claims_by_aspect[aspect].append(
							{
								"paper": title,
								"polarity": polarity,
								"evidence": clause.strip()[:220],
							}
						)

	contradictions: List[Dict[str, Any]] = []
	for aspect, claims in claims_by_aspect.items():
		positives = [c for c in claims if c["polarity"] == "positive"]
		negatives = [c for c in claims if c["polarity"] == "negative"]
		if positives and negatives:
			contradictions.append(
				{
					"aspect": aspect,
					"paper_a": positives[0]["paper"],
					"claim_a": positives[0]["evidence"],
					"paper_b": negatives[0]["paper"],
					"claim_b": negatives[0]["evidence"],
				}
			)

	return contradictions[:5]


def _claim_polarity(sentence: str) -> str:
	pos = bool(_POSITIVE_PAT.search(sentence))
	neg = bool(_NEGATIVE_PAT.search(sentence))
	if pos and not neg:
		return "positive"
	if neg and not pos:
		return "negative"
	if pos and neg:
		if re.search(r"\bno\s+improv(?:e|ement)\b", sentence):
			return "negative"
		return "positive"
	return "neutral"


def _build_hypotheses(gaps: List[str], core_themes: List[str], topic: str) -> List[str]:
	theme_text = core_themes[0] if core_themes else "identified core themes"
	hypotheses: List[str] = []

	for idx, gap in enumerate(gaps[:5], start=1):
		gap_core = re.sub(r"^In\s+.+?,\s+current literature shows\s+", "", gap).rstrip(".")
		hypotheses.append(
			f"H{idx}: Addressing {gap_core} in {topic} will significantly improve outcomes related to {theme_text}."
		)

	if not hypotheses:
		hypotheses = [
			f"H1: A targeted intervention in {topic} will outperform existing baselines across key evaluation metrics.",
			f"H2: Integrating robustness and fairness checks in {topic} will reduce performance variance across settings.",
		]
	return hypotheses


def _build_positioning_statement(gaps: List[str], core_themes: List[str], topic: str) -> str:
	top_gap = gaps[0].replace("In ", "").rstrip(".") if gaps else "fragmented evidence"
	theme = core_themes[0] if core_themes else "core literature themes"
	return (
		f"This study positions itself at the intersection of {topic} and {theme}, "
		f"specifically targeting {top_gap.lower()} through a measurable and reproducible methodology."
	)
