from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


_ASPECTS: Dict[str, List[str]] = {
    "accuracy": ["accuracy", "f1", "precision", "recall", "auc"],
    "efficiency": ["latency", "throughput", "runtime", "compute", "efficiency"],
    "privacy": ["privacy", "confidentiality", "anonym", "dp", "differential privacy"],
    "robustness": ["robust", "noise", "adversarial", "out-of-distribution", "ood"],
    "fairness": ["fair", "bias", "equity", "demographic"],
    "generalization": ["generaliz", "cross-domain", "transfer", "external validity"],
}

_POSITIVE_PAT = re.compile(r"\b(improv(?:e|ed|es|ement)|outperform(?:s|ed)?|increase(?:d)?)\b")
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

_GENERIC_THEME_LABELS = {
    "model", "models", "system", "systems", "approach", "approaches",
    "method", "methods", "data", "paper", "papers", "study", "studies",
    "llm", "llms", "task", "tasks",
}

# Human-readable descriptions for each gap type used in statements/hypotheses
_GAP_PRETTY: Dict[str, str] = {
    "generalization": "limited cross-domain generalization",
    "scalability": "insufficient large-scale validation",
    "privacy_security": "incomplete privacy and security analysis",
    "interpretability": "weak model interpretability evidence",
    "robustness": "insufficient robustness under noisy or adversarial settings",
    "fairness": "limited fairness and demographic bias evaluation",
    "real_world_validation": "lack of real-world deployment studies",
    "resource_constraints": "limited evaluation under low-resource or edge constraints",
    "long_term_evaluation": "missing longitudinal performance tracking",
    "theme_coverage": "uneven coverage across reported themes",
}

# Measurable hypothesis templates, one per gap type
_HYPOTHESIS_TEMPLATES: Dict[str, str] = {
    "generalization": (
        "H{idx}: A method addressing {gap_core} in {topic} will achieve ≥10% higher "
        "out-of-distribution accuracy than current baselines on held-out external datasets for {theme}."
    ),
    "privacy_security": (
        "H{idx}: Integrating privacy-preserving mechanisms to address {gap_core} in {topic} "
        "will reduce membership-inference advantage by ≥20% while keeping accuracy within 5% of "
        "non-private baselines for {theme}."
    ),
    "robustness": (
        "H{idx}: A robustness-targeted approach addressing {gap_core} in {topic} will improve "
        "accuracy-under-attack by ≥15% over standard baselines across noise and adversarial benchmarks for {theme}."
    ),
    "fairness": (
        "H{idx}: Applying fairness constraints to address {gap_core} in {topic} will reduce "
        "cross-demographic performance disparity by ≥10% on standard fairness metrics "
        "while maintaining overall utility for {theme}."
    ),
    "real_world_validation": (
        "H{idx}: Deploying and evaluating {topic} in a real-world setting will surface actionable "
        "failure modes and yield reproducible improvements (≥10% on task-relevant metrics) for {theme}."
    ),
    "scalability": (
        "H{idx}: A scalable architecture addressing {gap_core} in {topic} will scale to 10× "
        "the baseline node count with <20% communication-overhead increase while preserving model "
        "performance for {theme}."
    ),
    "interpretability": (
        "H{idx}: Introducing interpretability mechanisms to address {gap_core} in {topic} will "
        "raise human-evaluated explanation quality scores by ≥1 point (5-point scale) and increase "
        "practitioner adoption likelihood for {theme}."
    ),
    "resource_constraints": (
        "H{idx}: An efficient design targeting {gap_core} in {topic} will reduce peak "
        "memory/compute cost by ≥30% on edge hardware while keeping accuracy within 3% of "
        "the full-resource baseline for {theme}."
    ),
    "long_term_evaluation": (
        "H{idx}: A longitudinal evaluation protocol for {topic} will detect performance drift "
        "within 3 months of deployment and enable corrective interventions that restore accuracy "
        "by >10% over time for {theme}."
    ),
}

_HYPOTHESIS_FALLBACK = (
    "H{idx}: Directly addressing {gap_core} in {topic} will produce measurable improvements "
    "of >10% on the primary evaluation metric compared to existing baselines for {theme}."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_topic(topic: str) -> str:
    text = re.sub(r"\s+", " ", (topic or "")).strip(" \t\n\r:;,-")
    if not text:
        return "the study"
    if ":" in text:
        head, tail = text.split(":", 1)
        head = head.strip()
        tail = tail.strip()
        tail = re.sub(
            r"\b(and|with|for|to|of)\s+(models?|systems?|data|papers?|studies?)\b.*$",
            "", tail, flags=re.IGNORECASE,
        )
        tail = re.sub(r"\b(and|with|for|to|of)\s*$", "", tail, flags=re.IGNORECASE).strip(" \t\n\r:;,-")
        if tail:
            return f"{head}: {tail.title()}"
        return head
    return text.title() if text.isupper() else text


def _select_theme_label(core_themes: List[str], topic: str) -> str:
    """
    Return the first theme that is non-generic and not a substring/superset of the topic.
    Falls back to the topic's sub-specifier (after ':') or a neutral phrase.

    FIX: The original code had `continue; return clean` — the return was unreachable,
    so the function always fell through to "the target application domain".
    """
    topic_l = topic.lower().strip()
    for theme in core_themes:
        clean = re.sub(r"\s+", " ", str(theme or "")).strip(" \t\n\r:;,-")
        clean_l = clean.lower()
        if not clean:
            continue
        if clean_l in _GENERIC_THEME_LABELS:
            continue
        # Skip themes that are just a restatement of the topic
        if topic_l and (clean_l in topic_l or topic_l in clean_l):
            continue
        return clean  # ← was unreachable in the original; fixed by removing the stray `continue`

    # Secondary fallback: use the tail of a colon-separated topic
    if ":" in topic:
        _, tail = topic.split(":", 1)
        tail = tail.strip(" \t\n\r:;,-")
        if tail and tail.lower() not in _GENERIC_THEME_LABELS and tail.lower() != topic_l:
            return tail

    return "the target application domain"


def _looks_like_title_or_header(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) < 18:
        return True
    if len(clean.split()) < 6:
        return True
    letters = [ch for ch in clean if ch.isalpha()]
    if not letters:
        return True
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    if upper_ratio >= 0.65:
        return True
    if re.search(
        r"\b(arxiv|doi|http[s]?://|www\.|et al\.|figure\s+\d+|table\s+\d+|index\s+terms|running\s+title)\b",
        clean, re.IGNORECASE,
    ):
        return True
    if re.search(r"question\s+[ab]\s+is\s+better", clean, re.IGNORECASE):
        return True
    return False


def _extract_gap_core(gap: str, topic: str) -> str:
    """
    Strip the boilerplate prefix added by _build_gap_statements so hypotheses
    start with the actual gap phrase, not 'In X, current literature shows …'.
    """
    pattern = rf"^In\s+{re.escape(topic)}\s*,\s*current literature shows\s+"
    clean = re.sub(pattern, "", gap, flags=re.IGNORECASE).rstrip(".").strip()
    if clean:
        return clean
    # Generic fallback strip
    return re.sub(
        r"^In\s+.+?,\s+current literature shows\s+", "", gap, flags=re.IGNORECASE
    ).rstrip(".").strip()


def _gap_key_from_core(gap_core: str) -> str | None:
    """
    Map a gap phrase back to a template key by checking the pretty-label map
    and the raw key names. This replaces the fragile string-contains check in
    the original _build_hypotheses.
    """
    gap_lower = gap_core.lower()
    # Check pretty labels first (more readable descriptions)
    for key, label in _GAP_PRETTY.items():
        # Match any significant word from the label
        label_words = [w for w in label.split() if len(w) > 4]
        if any(word in gap_lower for word in label_words):
            return key
    # Check raw key names as last resort
    for key in _HYPOTHESIS_TEMPLATES:
        if key.replace("_", " ") in gap_lower or key in gap_lower:
            return key
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    topic_label = _normalize_topic(research_topic)
    theme_label = _select_theme_label(themes, topic_label)

    corpus_parts: List[str] = []
    for paper in papers:
        corpus_parts.append(_paper_to_text(paper))
    corpus_parts.extend(themes)
    corpus_text = "\n".join(corpus_parts).lower()

    gap_scores = _compute_gap_scores(corpus_text, citation_map, themes)
    identified_gaps = _build_gap_statements(gap_scores, topic_label)
    contradiction_pairs = _find_contradictions(papers)
    hypotheses = _build_hypotheses(identified_gaps, [theme_label], topic_label)
    positioning_statement = _build_positioning_statement(identified_gaps, [theme_label], topic_label)

    return {
        "identified_gaps": identified_gaps,
        "gap_frequency_scores": gap_scores,
        "hypotheses": hypotheses,
        "positioning_statement": positioning_statement,
        "contradiction_pairs": contradiction_pairs,
    }


# ---------------------------------------------------------------------------
# Internal pipeline steps
# ---------------------------------------------------------------------------

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

    # Theme coverage adjustment: low-citation themes usually indicate potential gaps
    for theme in core_themes:
        theme_l = theme.lower()
        if theme_l and theme_l in corpus_text:
            counts["theme_coverage"] += 1

    if isinstance(citation_map, dict):
        for key, value in citation_map.items():
            if _is_low_citation(value):
                counts["theme_coverage"] += 2
            if isinstance(key, str) and re.search(
                r"contradict|conflict|inconsisten", key, re.IGNORECASE
            ):
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
    """
    Build concrete gap statements from the top-scored gap types.

    FIX: The original used a hard-coded pretty-label dict defined inline; now it
    references the module-level _GAP_PRETTY so templates stay in sync with
    hypothesis templates.
    """
    top_keys = list(gap_scores.keys())[:5] if gap_scores else []
    gaps: List[str] = []
    for key in top_keys:
        phrase = _GAP_PRETTY.get(key, key.replace("_", " "))
        gaps.append(f"In {topic}, current literature shows {phrase}.")

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
                if not sent_l or _looks_like_title_or_header(clause):
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
        pair: tuple | None = None
        for pos in positives:
            for neg in negatives:
                if pos["paper"] != neg["paper"]:
                    pair = (pos, neg)
                    break
            if pair is not None:
                break
        if pair is not None:
            contradictions.append(
                {
                    "aspect": aspect,
                    "paper_a": pair[0]["paper"],
                    "claim_a": pair[0]["evidence"],
                    "paper_b": pair[1]["paper"],
                    "claim_b": pair[1]["evidence"],
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
        return "negative" if re.search(r"\bno\s+improv(?:e|ement)\b", sentence) else "positive"
    return "neutral"


def _build_hypotheses(gaps: List[str], core_themes: List[str], topic: str) -> List[str]:
    """
    Generate one concrete, measurable hypothesis per identified gap.

    FIX 1 — Theme resolution: now calls the corrected _select_theme_label so
             'theme_display' is actually populated from core_themes.
    FIX 2 — Template matching: delegates to _gap_key_from_core which checks
             both pretty labels and raw keys, replacing the fragile inline loop
             that almost never matched.
    FIX 3 — String formatting: templates are stored with {{ }} escapes and
             formatted with .format() so variable injection is explicit.
    """
    theme_label = _select_theme_label(core_themes, topic)

    # If the selected theme is essentially the same as the topic, use a neutral phrase
    if theme_label and topic and (
        theme_label.strip().lower() in topic.strip().lower()
        or topic.strip().lower() in theme_label.strip().lower()
    ):
        theme_display = "this application domain"
    else:
        theme_display = theme_label

    hypotheses: List[str] = []
    for idx, gap in enumerate(gaps[:5], start=1):
        gap_core = _extract_gap_core(gap, topic)
        gap_key = _gap_key_from_core(gap_core)
        template = _HYPOTHESIS_TEMPLATES.get(gap_key, _HYPOTHESIS_FALLBACK) if gap_key else _HYPOTHESIS_FALLBACK

        hypotheses.append(
            template.format(idx=idx, gap_core=gap_core, topic=topic, theme=theme_display)
        )

    if not hypotheses:
        hypotheses = [
            f"H1: A targeted intervention in {topic} will outperform existing baselines on clearly defined metrics.",
            f"H2: Integrating robustness and fairness checks in {topic} will reduce performance variance across settings.",
        ]
    return hypotheses


def _build_positioning_statement(gaps: List[str], core_themes: List[str], topic: str) -> str:
    """
    Build a one-sentence positioning statement.

    FIX: The original short-circuited to a vague statement whenever topic and theme
    overlapped even slightly. Now it always names the primary gap explicitly, and
    only omits the theme phrase when the overlap is an exact/near-exact match.
    """
    top_gap = _extract_gap_core(gaps[0], topic) if gaps else "fragmented evidence and limited validation"
    theme = _select_theme_label(core_themes, topic)

    topic_l = topic.strip().lower()
    theme_l = theme.strip().lower()
    themes_overlap = (theme_l == topic_l or theme_l == "the target application domain")

    if themes_overlap:
        return (
            f"This study addresses {topic} by specifically targeting {top_gap} "
            f"through a measurable and reproducible methodology."
        )

    return (
        f"This study positions itself at the intersection of {topic} and {theme}, "
        f"specifically targeting {top_gap} through a measurable and reproducible methodology."
    )