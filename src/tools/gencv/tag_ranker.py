"""
gencv/tag_ranker.py

Deterministic JD-relevance ranking for tags. Shared by build_skill_entries
(top-10 per group) and build_project_entries (top-3 tools per project).

No LLM calls: a tag's relevance score is how many times its tag_name (and
any aliases, once populated) appears as a whole token / whole phrase in the
JD text. Deterministic and reproducible. Ties (incl. all-zero scores) break
alphabetically by tag_name, so the order is always fully determined.
"""

from __future__ import annotations

import re

# Keep +, #, . inside tokens so "C++", "C#", ".NET", "Node.js" survive.
_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _normalize(text: str) -> tuple[list[str], str]:
    lowered = (text or "").lower()
    tokens = _TOKEN_RE.findall(lowered)
    return tokens, " ".join(tokens)


def _tag_score(surface_forms, jd_tokens, jd_joined, token_counts) -> int:
    score = 0
    for form in surface_forms:
        form_tokens = _TOKEN_RE.findall(form.lower().strip())
        if not form_tokens:
            continue
        if len(form_tokens) == 1:
            score += token_counts.get(form_tokens[0], 0)
        else:
            phrase = " ".join(form_tokens)
            pattern = r"(?<!\S)" + re.escape(phrase) + r"(?!\S)"
            score += len(re.findall(pattern, jd_joined))
    return score


def rank_tags_by_jd(tag_names, job_description, aliases_map=None):
    """Ranks tag_names by JD relevance, most relevant first.
    Returns list of (tag_name, score) sorted by (-score, tag_name).
    aliases_map: optional {tag_name: [alias, ...]}; safe to omit while
    tag.aliases is still empty."""
    jd_tokens, jd_joined = _normalize(job_description)
    token_counts = {}
    for t in jd_tokens:
        token_counts[t] = token_counts.get(t, 0) + 1

    aliases_map = aliases_map or {}
    scored = []
    for name in tag_names:
        forms = [name] + list(aliases_map.get(name, []))
        scored.append((name, _tag_score(forms, jd_tokens, jd_joined, token_counts)))

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def top_k_tags_by_jd(tag_names, job_description, k, aliases_map=None):
    """Rank then take the top k tag_name strings only. Used by
    build_project_entries (k=3) and per-group top-10 trimming."""
    ranked = rank_tags_by_jd(tag_names, job_description, aliases_map=aliases_map)
    return [name for name, _ in ranked[:k]]