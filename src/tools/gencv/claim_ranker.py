"""
claim_ranker.py
Version: v1 (2026-08-16)

Solves the "same-project claim redundancy" problem: when a single project
has more than MAX_CLAIMS_PER_PROJECT relevant claims (per
experience_matcher.select_claims()'s JD-relevance judgment), simply
keeping the top-N by JD-fit tends to collapse toward claims that are ALL
very similar to each other -- because "most JD-relevant" and "most
similar to each other" are correlated when a project's claims cluster
around one dominant capability. Picking by relevance alone, then asking
experience_rewriter.py to "diversify the angle" afterward, is asking the
rewrite step to manufacture differences that don't exist in the
underlying facts -- which conflicts with the project's core safety
principle (LLM may not introduce claims/framing not supported by source
material).

This module instead selects for CAPABILITY DIVERSITY FIRST, with JD
relevance as a gate (not a ranking axis) that has already been satisfied
by experience_matcher.select_claims() before this module ever runs. The
selected claims are then genuinely heterogeneous BEFORE
experience_rewriter.py touches them, so its "diversify the angle"
instruction has real factual grounding to work with instead of being
asked to invent variety.

Design decisions (per user confirmation, 2026-08-16):
1. MAX_CLAIMS_PER_PROJECT = 2 (hard cap set by user). Below or at this
threshold, every candidate claim is kept -- no ranking/dropping logic
runs at all, and no LLM call is made.
2. Primary signal: Jaccard similarity over each claim's tag_name set
(via claim_tag -> tag), NOT tag_category (skill/domain/job_function is
too coarse -- two claims can share a category but cover completely
different skills). This requires zero LLM calls and is deterministic
(same input always produces the same grouping, unlike an LLM call at
nonzero temperature). Confirmed viable because claims typically carry
5-6 tags each in this user's data -- tag coverage is NOT a fallback
edge case here, it's the default, reliable path.
3. Greedy minimum-average-similarity selection: repeatedly picks the
claim whose average Jaccard similarity to the already-selected set is
lowest, which tends toward maximally different claims rather than an
arbitrary top-N cut. This is a similarity-driven heuristic, not an
optimal set-cover solution -- deliberately simple, since
MAX_CLAIMS_PER_PROJECT is only 2 (picking 2 out of a small candidate
pool doesn't need a heavier combinatorial approach).
4. LLM fallback ONLY triggers when tag coverage is insufficient to make
a confident similarity judgment -- specifically, when at least one
candidate claim in the project has zero tags. Given the user's
confirmed 5-6 tags/claim norm, this path will rarely fire in practice,
but must not silently mis-handle the rare untagged claim by assuming
either "unique" or "duplicate" without basis. The LLM fallback reads
full Problem/Action/Result text (not just tags) for the whole
project's candidate claims, and is explicitly instructed to fall back
to "select by overall JD fit" if the project's claims are genuinely
single-faceted (no real diversity to find) -- it must not invent
capability differences that don't exist in the source material.
5. Lightweight (not per-claim) confirmation gate: shows the user which
claims were kept vs. dropped for each affected project, in one summary
block per project, with an explicit note that dropping is SCOPED TO
THIS GENERATION RUN ONLY -- the claim itself is untouched in the
database and may be selected again for a different job description.
This is deliberately lighter than experience_rewriter.py's per-bullet
_confirm_one() gate (that gate reviews newly-generated TEXT; this gate
reviews a SELECTION among already-approved claims, a lower-stakes
decision), but still gives the user visibility and an override option,
consistent with the project's "no silent batch decisions" principle.

Runs in pipeline.py (v9+) between experience_matcher.select_claims() and
every downstream step -- takes the JD-relevant claim_ids
experience_matcher already produced, narrows any over-populated project
down to MAX_CLAIMS_PER_PROJECT, and passes the resulting (possibly
smaller) claim_ids list forward unchanged in shape (still a flat
list[str]) so downstream calls don't need to change.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from core.db import get_session
from core.llm_client import get_llm_client
from gencv.experience_matcher import fetch_experience_contexts, fetch_tags_for_claims

MAX_CLAIMS_PER_PROJECT = 2

RANKER_SYSTEM_PROMPT = """You are a resume content curator. You will be given a job description and
several candidate experience records that ALL belong to the SAME project
and have ALREADY been judged relevant to this job -- your task is NOT to
re-judge relevance, it is to pick which ones to keep when there are more
candidates than the resume has room for.

Your task: identify the core CAPABILITY each record demonstrates (e.g.
technical depth, cross-team collaboration, data-driven decision-making,
ownership/leadership, process improvement -- these are illustrative only,
determine the actual capability freeing yourself from this specific list),
then select exactly {max_claims} record(s) whose capabilities are as
DIFFERENT from each other as possible, so the resume shows the candidate's
range rather than repeating the same strength.

Strict rules (must follow):
1. Every record given to you has already passed a relevance check --
do not exclude a record for being "less relevant"; your only basis for
choosing is capability diversity.
2. If the candidate records are GENUINELY single-faceted -- i.e. they all
demonstrate essentially the same capability with no real diversity to
find -- say so explicitly and select the {max_claims} record(s) that
best represent the project's overall contribution instead. Do NOT
invent or exaggerate a capability distinction that isn't actually
supported by the record's content -- fabricating diversity would
violate this project's core rule against introducing unsupported
framing.
3. Do not add any fact, number, or claim not already present in the
records given.
4. Output must be in English, regardless of input language.

Output format: first line is either "DIVERSE" or "SINGLE-FACETED" (your
assessment). Second line is the selected record number(s), comma-separated
(exactly {max_claims}, or fewer only if fewer than {max_claims} records
were given). No other text, no explanation beyond these two lines."""

RANKER_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Candidate records from the same project ({count} total, all already JD-relevant)
{records_block}

Output your assessment (DIVERSE or SINGLE-FACETED) and the {max_claims} selected record number(s):"""


def _group_claims_by_project(
    claim_ids: list[str], contexts: dict[str, dict]
) -> dict[str, list[str]]:
    """claim_id 清單依 project_id 分組。project_id 為 None（理論上不該發生，
    因為每個 claim 都透過 action->problem->project 鏈掛在某個 project 下）的
    claim 會各自獨立成組，視為「無須分組取捨」，直接保留。"""
    grouped: dict[str, list[str]] = defaultdict(list)
    for claim_id in claim_ids:
        ctx = contexts.get(claim_id)
        project_id = ctx.get("project_id") if ctx else None
        key = project_id if project_id else f"__no_project__{claim_id}"
        grouped[key].append(claim_id)
    return grouped


def _jaccard_similarity(tags_a: set[str], tags_b: set[str]) -> float:
    """標準 Jaccard 相似度：|交集| / |聯集|。兩者皆為空集合時視為 0.0
    （無法判斷相似度，保守處理，不當作「完全相同」或「完全不同」）。"""
    if not tags_a and not tags_b:
        return 0.0
    union = tags_a | tags_b
    if not union:
        return 0.0
    return len(tags_a & tags_b) / len(union)


def _select_by_tag_diversity(
    candidate_ids: list[str], tags_map: dict[str, list[str]], max_claims: int
) -> list[str]:
    """貪婪最小平均相似度選擇：從候選集合中，重複挑選「跟已選集合平均
    Jaccard 相似度最低」的 claim，直到選滿 max_claims 個。傾向選出彼此
    差異最大的組合，而非單純依單一分數排序取前 N。

    第一個被選中的 claim：整體跟其他所有候選的平均相似度最低者
    （代表「最獨特」的起始點）。
    """
    tag_sets = {cid: set(tags_map.get(cid, [])) for cid in candidate_ids}

    if len(candidate_ids) <= max_claims:
        return list(candidate_ids)

    pairwise: dict[tuple[str, str], float] = {}
    for a, b in combinations(candidate_ids, 2):
        sim = _jaccard_similarity(tag_sets[a], tag_sets[b])
        pairwise[(a, b)] = sim
        pairwise[(b, a)] = sim

    def avg_similarity_to(cid: str, others: list[str]) -> float:
        if not others:
            return 0.0
        return sum(pairwise[(cid, o)] for o in others) / len(others)

    remaining = list(candidate_ids)
    first = min(remaining, key=lambda cid: avg_similarity_to(cid, [c for c in remaining if c != cid]))
    selected = [first]
    remaining.remove(first)

    while len(selected) < max_claims and remaining:
        next_pick = min(remaining, key=lambda cid: avg_similarity_to(cid, selected))
        selected.append(next_pick)
        remaining.remove(next_pick)

    return selected


def _has_full_tag_coverage(candidate_ids: list[str], tags_map: dict[str, list[str]]) -> bool:
    """True 代表每一筆候選 claim 都至少有一個 tag -- 足以信任 Jaccard 判斷。
    只要有任何一筆 claim 是零 tag，就整組退回 LLM 判斷（不對那一筆單獨做
    無依據的「跟誰都不像」或「跟誰都像」假設）。"""
    return all(tags_map.get(cid) for cid in candidate_ids)


def _build_ranker_prompt(
    candidate_ids: list[str],
    contexts: dict[str, dict],
    job_description: str,
    max_claims: int,
) -> tuple[str, dict[int, str]]:
    index_to_claim_id: dict[int, str] = {}
    lines = []
    for i, claim_id in enumerate(candidate_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        index_to_claim_id[i] = claim_id
        lines.append(
            f"[{i}] Problem: {ctx.get('problem_description') or '(not provided)'}\n"
            f"    Action: {ctx.get('action_description') or '(not provided)'}\n"
            f"    Result: {ctx.get('claim_description') or '(not provided)'}"
        )
    records_block = "\n\n".join(lines)
    user_prompt = RANKER_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(index_to_claim_id),
        records_block=records_block,
        max_claims=max_claims,
    )
    return user_prompt, index_to_claim_id


def _select_by_llm_judgment(
    candidate_ids: list[str],
    contexts: dict[str, dict],
    job_description: str,
    max_claims: int,
) -> tuple[list[str], bool]:
    """LLM fallback，僅在 tag 覆蓋率不足時觸發。回傳 (選中的 claim_id 清單,
    是否為 single-faceted 判斷)。"""
    user_prompt, index_to_claim_id = _build_ranker_prompt(
        candidate_ids, contexts, job_description, max_claims
    )
    if not index_to_claim_id:
        return candidate_ids[:max_claims], False

    response = get_llm_client().generate_text(
        system_prompt=RANKER_SYSTEM_PROMPT.format(max_claims=max_claims),
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=100,
    )

    lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
    is_single_faceted = bool(lines) and "SINGLE-FACETED" in lines[0].upper()

    indices: list[int] = []
    if len(lines) >= 2:
        indices = [int(n) for n in re.findall(r"\d+", lines[1])]
    elif lines:
        indices = [int(n) for n in re.findall(r"\d+", lines[0])]

    selected = [index_to_claim_id[i] for i in indices if i in index_to_claim_id][:max_claims]
    if not selected:
        selected = candidate_ids[:max_claims]

    return selected, is_single_faceted


def _confirm_project_selection(
    project_label: str,
    kept_ids: list[str],
    dropped_ids: list[str],
    contexts: dict[str, dict],
    method_note: str,
) -> list[str]:
    """輕量確認關卡（非逐條 Enter）：一次性顯示某 project 的取捨結果，
    讓使用者看一眼即可，可覆寫。捨棄僅限本次生成，不影響資料庫本身。"""
    if not dropped_ids:
        return kept_ids

    print(f"\n=== Project \"{project_label}\": narrowed from {len(kept_ids) + len(dropped_ids)} to {len(kept_ids)} claim(s) ({method_note}) ===")
    for cid in kept_ids:
        desc = contexts.get(cid, {}).get("claim_description", "")
        print(f"  [KEEP]   {desc}")
    for cid in dropped_ids:
        desc = contexts.get(cid, {}).get("claim_description", "")
        print(f"  [DROP]   {desc}")
    print("  (Dropped claims are excluded from THIS resume only -- they remain in your "
          "database and may be selected again for a different job description.)")

    choice = input(
        "  Accept this selection? (Enter to accept / type comma-separated numbers "
        "from the full list above to override, using 1-based position in KEEP+DROP order): "
    ).strip()

    if not choice:
        return kept_ids

    all_ids = kept_ids + dropped_ids
    try:
        indices = [int(x.strip()) for x in choice.split(",") if x.strip()]
        override = [all_ids[i - 1] for i in indices]
        print(f"  -> Using manually selected: {len(override)} claim(s)")
        return override
    except (ValueError, IndexError):
        print("  [Warning] Could not parse override -- keeping the original selection")
        return kept_ids


def rank_and_narrow_claims(claim_ids: list[str], job_description: str) -> list[str]:
    """主入口。輸入 experience_matcher.select_claims() 已判斷 JD-relevant 的
    claim_id 清單，依 project 分組，任何超過 MAX_CLAIMS_PER_PROJECT 的 project
    會被narrowed down -- 優先依 tag Jaccard 相似度做多樣性選擇（零額外 LLM
    呼叫），tag 覆蓋率不足時才退回 LLM 判斷。回傳結構不變（仍是 flat
    list[str]），供 pipeline.py 直接接續原有呼叫。
    """
    if not claim_ids:
        return []

    with get_session() as session:
        contexts = fetch_experience_contexts(session, claim_ids)
        tags_map = fetch_tags_for_claims(session, claim_ids)

    grouped = _group_claims_by_project(claim_ids, contexts)

    final_ids: list[str] = []

    for project_key, group_claim_ids in grouped.items():
        if len(group_claim_ids) <= MAX_CLAIMS_PER_PROJECT:
            final_ids.extend(group_claim_ids)
            continue

        project_label = contexts.get(group_claim_ids[0], {}).get("project_name") or project_key

        if _has_full_tag_coverage(group_claim_ids, tags_map):
            kept = _select_by_tag_diversity(group_claim_ids, tags_map, MAX_CLAIMS_PER_PROJECT)
            method_note = "tag-based diversity selection"
        else:
            kept, is_single_faceted = _select_by_llm_judgment(
                group_claim_ids, contexts, job_description, MAX_CLAIMS_PER_PROJECT
            )
            method_note = "LLM judgment: single-faceted, selected by overall fit" if is_single_faceted else "LLM judgment: capability diversity"

        dropped = [cid for cid in group_claim_ids if cid not in kept]
        confirmed = _confirm_project_selection(project_label, kept, dropped, contexts, method_note)
        final_ids.extend(confirmed)

    return final_ids
