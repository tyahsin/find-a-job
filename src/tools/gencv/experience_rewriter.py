"""
experience_rewriter.py
Version: v4 (2026-08-16)

Changelog:
- v4 (2026-08-16): Added PROJECT-GROUPED batch awareness to address the
"same-project bullets read as repetitive" problem. Previously,
_build_batch_prompt() listed every claim flat, in whatever order
claim_ids arrived in -- the LLM rewrote each record in isolation with
zero visibility into any OTHER claim selected for the same project, so
two claims from the same project could easily converge on similar
verbs/phrasing even when their underlying facts differed.

Fix: claim_ids are now grouped by project_id (via each claim's context)
before building the prompt. Records are still rewritten in ONE single
batch LLM call (no added LLM cost), but the prompt is now organized into
explicit per-project blocks, and any project with more than one claim in
the batch gets an explicit instruction: rewrite these bullets so each one
foregrounds a DIFFERENT capability angle (e.g. technical depth vs.
cross-team collaboration vs. data-driven decisions), based on what the
source material actually supports -- never invent a distinction the
facts don't back up.

This is deliberately SEPARATE from, and runs AFTER, claim_ranker.py
(pipeline.py v9+) -- claim_ranker.py's job is to narrow an over-populated
project down to MAX_CLAIMS_PER_PROJECT genuinely-different claims BEFORE
this file ever sees them. This module's diversification instruction is a
presentation-layer reinforcement on top of already-diverse source
material, not a substitute for picking diverse claims in the first place
-- asking this step to manufacture diversity from claims that are
factually identical would violate the project's core "no unsupported
framing" rule. If a project's claims genuinely share the same capability
(e.g. claim_ranker.py assessed them SINGLE-FACETED), this module does not
force artificial variation -- see rule 6 below.
- v3 (2026-08-16): Rewrote REWRITE_SYSTEM_PROMPT to fix bullets that were
too long and unfocused: added a ~20-25 word length guideline (judged by
rendered line count, not sentence/period count), required the bullet to
lead with a strong action verb, and made Situation/Problem context
omitted by default (only kept as a short clause at the START when it
carries real scope/difficulty signal -- never moved to the end).
- v2 (2026-08-15): Merged STAR-fusion (background + problem + action +
result -> one sentence) with JD-tailoring rewrite into ONE single batch
LLM call per generate_cv() run, followed by ONE mandatory human
confirmation gate per bullet (not two separate passes).
- v1 (2026-08-10, inferred): Initial two-pass STAR fusion + JD tailoring.

Safety design principles (unchanged since v2):
1. Metric numbers are NEVER touched by the LLM here.
2. governance.notes_for_llm is respected as a hard constraint.
3. The LLM may only reorganize/condense the four source fields it's given
-- no new facts, numbers, scope, or degree of achievement.
4. Every bullet requires human confirmation (Enter to accept / type new
text to override) -- no silent batch acceptance.
5. (v4) Capability-angle diversification is presentation-only -- it may
reframe HOW an already-true fact is described, never invent a
capability or achievement the source material doesn't support.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from collections import defaultdict

from core.db import get_session, GovernanceORM
from core.llm_client import get_llm_client
from gencv.experience_matcher import fetch_experience_contexts

REWRITE_SYSTEM_PROMPT = """You are a resume bullet-point writer. You will be given several experience
records, each with four pieces of source material -- project background,
problem, action, and result -- plus the target job description and any
per-record operating notes. Records are organized into PROJECT GROUPS;
records in the same group came from the same underlying project. Your
task is to fuse the relevant source fields into ONE single, fluent
bullet-point sentence per record, AND tailor the wording to align with
the target job description's terminology and emphasis.

Strict rules (must follow):
1. Do not add any concrete fact, number, scope, or degree of achievement
that is not already present in the four source fields you are given.
2. Do not change the direction or strength of any claim -- if the source
says "supported", do not upgrade it to "led"; if a result is described
modestly, do not amplify it.
3. LENGTH: keep each bullet to approximately 20-25 words -- roughly one
to two lines at typical resume column width. This is a WORD-COUNT
guideline based on rendered line length, NOT a sentence-count or
period-count rule -- a single sentence that runs long is just as much
a violation as two short sentences that together exceed the limit.
Cut adjectives, process detail, and any information that doesn't help
a reader judge WHAT was done and HOW WELL, before cutting anything
that affects the actual claim.
4. STRUCTURE: start the bullet with a strong action verb (e.g.
"Redesigned...", "Implemented...", "Streamlined..."), not a passive
construction (avoid "was tasked with", "was responsible for") and not
a Situation/Problem clause. Action and Result are the primary content
of the bullet.
5. The Situation/Problem context is OMITTED by default. Only include it
when the problem itself signals meaningful scope, scale, or difficulty
that a reader would otherwise miss (e.g. "a legacy system with no test
coverage"). When included, keep it to a short clause at the START of
the bullet (e.g. "To reduce onboarding friction, redesigned...") --
never place it at the end of the bullet as a trailing "...to address X"
clause; resumes are scanned left-to-right and the action verb must be
the first thing a reader encounters.
6. CAPABILITY DIVERSITY WITHIN A PROJECT GROUP: when a project group
contains MORE THAN ONE record, make sure each bullet in that group
foregrounds a DIFFERENT capability angle (e.g. technical depth,
cross-team collaboration, data-driven decision-making, ownership,
process improvement -- these are illustrative only, use whatever
angle the source material actually supports). This means choosing
DIFFERENT action verbs and DIFFERENT emphasis for each record in the
group, based on what its own source material actually emphasizes --
never invent or exaggerate a capability distinction that isn't
genuinely supported by that record's own problem/action/result text.
If the records in a group are genuinely about the same capability with
no real distinction to draw out, do NOT force artificial variation --
write each one accurately and concisely instead; a fabricated
distinction would violate rule 1 and rule 2 above.
7. You may tailor WORD CHOICE to better match the job description's
terminology and emphasis -- but you may not restructure the sentence's
subject/responsibility (see rule 2) or introduce claims the source
material doesn't support.
8. Do NOT touch, reformat, or remove any metric/number mentioned in the
result field -- leave metric text exactly as given; a separate process
formats numbers afterward.
9. You must obey any per-record operating notes given (notes_for_llm) as
a hard constraint on top of the rules above.
10. Concise, resume-appropriate register -- no filler words, no
first-person pronouns ("I", "my").
11. If any of the four source fields is empty or not provided, write the
sentence using only the fields that ARE provided -- do not invent a
substitute for a missing field.
12. Output must be in English, regardless of the input language.

Output format: one fused, JD-tailored bullet sentence per record, in the
same order as given, separated by the delimiter "|||" (three pipe
characters). No numbering, no preamble, no explanation -- output ONLY the
sentences separated by |||."""

REWRITE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Experience records to fuse and tailor ({count} total, organized by project group)
{records_block}

Output {count} fused, JD-tailored STAR bullet sentences, in the same order as given above, separated by |||:"""


def _fetch_governance_notes(session, claim_ids: list[str]) -> dict[str, str]:
    """claim_id -> notes_for_llm 映射，用於改寫時的硬性約束。"""
    if not claim_ids:
        return {}
    import uuid
    uuid_ids = [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]
    rows = session.query(GovernanceORM).filter(GovernanceORM.claim_id.in_(uuid_ids)).all()
    return {str(g.claim_id): (g.notes_for_llm or "none") for g in rows}


def _group_claims_by_project(
    claim_ids: list[str], contexts: dict[str, dict]
) -> list[tuple[str, list[str]]]:
    """(v4) 依 project_id 分組，保留原始 claim_ids 的順序（同組內部順序、
    組與組之間的順序都沿用輸入順序，不重新排序），回傳
    [(project_label, [claim_id, ...]), ...] 的順序清單，而非 dict --
    因為輸出的 bullet 順序必須跟輸入 claim_ids 順序一致（下游
    resume_builder.py 依 claim_id 對應 description_overrides，順序本身
    不影響正確性，但維持原順序方便除錯與人工確認時的可預期性）。

    project_id 為 None（理論上不該發生）的 claim 各自獨立成組。
    """
    order: list[str] = []
    groups: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, str] = {}

    for claim_id in claim_ids:
        ctx = contexts.get(claim_id)
        project_id = ctx.get("project_id") if ctx else None
        key = project_id if project_id else f"__no_project__{claim_id}"
        if key not in groups:
            order.append(key)
            labels[key] = (ctx.get("project_name") if ctx else None) or "Unlabeled project"
        groups[key].append(claim_id)

    return [(labels[key], groups[key]) for key in order]


def _build_batch_prompt(
    claim_ids: list[str],
    contexts: dict[str, dict],
    notes_map: dict[str, str],
    job_description: str,
) -> tuple[str, list[str]]:
    """(v4) 組出單一批次 prompt，依 project 分組列出所有候選經歷的四段來源
    文字，組內超過一筆時附加多樣化指示。回傳 (prompt, ordered_claim_ids)
    -- ordered_claim_ids 的順序即為輸出句子要對應回的 claim_id 順序。
    """
    project_groups = _group_claims_by_project(claim_ids, contexts)

    ordered_claim_ids: list[str] = []
    record_number = 0
    blocks: list[str] = []

    for project_label, group_claim_ids in project_groups:
        group_lines = []
        group_record_numbers: list[int] = []

        for claim_id in group_claim_ids:
            ctx = contexts.get(claim_id)
            if not ctx:
                continue
            record_number += 1
            ordered_claim_ids.append(claim_id)
            group_record_numbers.append(record_number)
            notes_for_llm = notes_map.get(claim_id, "none")
            group_lines.append(
                f"  [Record {record_number}]\n"
                f"    Background: {ctx.get('project_context') or '(not provided)'}\n"
                f"    Problem: {ctx.get('problem_description') or '(not provided)'}\n"
                f"    Action: {ctx.get('action_description') or '(not provided)'}\n"
                f"    Result: {ctx.get('claim_description') or '(not provided)'}\n"
                f"    Operating notes: {notes_for_llm}"
            )

        if not group_lines:
            continue

        group_header = f"# Project group: {project_label} ({len(group_record_numbers)} record(s))"
        if len(group_record_numbers) > 1:
            group_header += (
                f"\n  NOTE: Records {', '.join(str(n) for n in group_record_numbers)} are from the "
                f"SAME project -- apply rule 6 (capability diversity within a project group)."
            )
        blocks.append(group_header + "\n" + "\n\n".join(group_lines))

    records_block = "\n\n".join(blocks)
    user_prompt = REWRITE_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(ordered_claim_ids),
        records_block=records_block,
    )
    return user_prompt, ordered_claim_ids


def _parse_batch_response(raw_response: str, expected_count: int) -> list[str]:
    """把 ||| 分隔的回應切回各條句子。"""
    parts = [p.strip() for p in raw_response.split("|||")]
    if len(parts) < expected_count:
        parts += [""] * (expected_count - len(parts))
    return parts[:expected_count]


def _draft_fused_bullets(
    claim_ids: list[str],
    contexts: dict[str, dict],
    notes_map: dict[str, str],
    job_description: str,
) -> dict[str, str]:
    """單次批次 LLM 呼叫（v4: 現在依 project 分組建構 prompt）。回傳
    {claim_id: 融合且已貼近JD的句子}。"""
    if not claim_ids:
        return {}

    user_prompt, ordered_claim_ids = _build_batch_prompt(
        claim_ids, contexts, notes_map, job_description
    )
    if not ordered_claim_ids:
        return {}

    response = get_llm_client().generate_text(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=200 * len(ordered_claim_ids),
    )

    sentences = _parse_batch_response(response.text, len(ordered_claim_ids))
    return dict(zip(ordered_claim_ids, sentences))


def _confirm_one(
    index: int, total: int, org_label: str, title_label: str,
    original: str, drafted: str,
) -> str:
    """逐條顯示原始成果描述 vs 融合且已貼近JD的完整句子，讓使用者決定用哪一版。"""
    print(f"\n[{index}/{total}] [{org_label} - {title_label}]")

    if not drafted.strip():
        print("  [Notice] LLM did not produce a usable fused sentence for this record.")
        print(f"  Falling back to original result text: {original}")
        choice = input("  Accept the fallback? (Enter to accept / type new text to override): ").strip()
        return choice if choice else original

    print(f"  Original result only: {original}")
    print(f"  Fused & JD-tailored bullet: {drafted}")

    if original.strip() == drafted.strip():
        print("  (Fused result is identical to the original -- using as-is)")
        return original

    choice = input("  Accept this bullet? (Enter to accept / type new text to override): ").strip()
    if choice:
        print("  -> Using manually entered text")
        return choice
    print("  -> Using fused & tailored bullet")
    return drafted


def rewrite_and_confirm(claim_ids: list[str], job_description: str) -> dict[str, str]:
    """對每一筆 claim 做「背景+問題+行動+成果」融合 + 貼近 JD 用語改寫，單次批次
    LLM 呼叫（v4: 依 project 分組，同組多筆時要求能力面向多樣化），逐條要求
    人工確認，回傳最終要用的完整句子文字。

    Note: claim_ids 通常是先經過 claim_ranker.rank_and_narrow_claims()
    處理過的清單（pipeline.py v9+ 已接入）-- 本函式的多樣化指示是呈現層
    優化，假設同 project 底下留到這裡的 claim 本身已經有事實層面的差異。
    """
    with get_session() as session:
        contexts = fetch_experience_contexts(session, claim_ids)
        notes_map = _fetch_governance_notes(session, claim_ids)

    print(f"\n=== Fusing background/problem/action/result + tailoring to JD for {len(claim_ids)} record(s) ===")

    drafts = _draft_fused_bullets(claim_ids, contexts, notes_map, job_description)

    final_descriptions: dict[str, str] = {}
    for i, claim_id in enumerate(claim_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue

        org_label = ctx.get("org_name") or "No organization"
        title_label = ctx.get("title") or ""
        original = ctx.get("claim_description") or ""
        drafted = drafts.get(claim_id, "")

        final_descriptions[claim_id] = _confirm_one(
            i, len(claim_ids), org_label, title_label, original, drafted
        )

    print("=" * 50)
    return final_descriptions
