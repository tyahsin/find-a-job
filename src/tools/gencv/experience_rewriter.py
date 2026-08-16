"""
experience_rewriter.py
Version: v2 (2026-08-15)

Merged STAR-fusion (background + problem + action + result -> one
sentence) with JD-tailoring rewrite into ONE single batch LLM call per
generate_cv() run, followed by ONE mandatory human confirmation gate per
bullet (not two separate passes).

Safety design principles:
1. Metric numbers are NEVER touched by the LLM here.
2. governance.notes_for_llm is respected as a hard constraint.
3. The LLM may only reorganize/condense the four source fields it's given
   -- no new facts, numbers, scope, or degree of achievement.
4. Every bullet requires human confirmation (Enter to accept / type new
   text to override) -- no silent batch acceptance.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from core.db import get_session, GovernanceORM
from core.llm_client import get_llm_client
from gencv.experience_matcher import fetch_experience_contexts

REWRITE_SYSTEM_PROMPT = """You are a resume bullet-point writer. You will be given several experience
records, each with four pieces of source material -- project background,
problem, action, and result -- plus the target job description and any
per-record operating notes. Your task is to fuse the four source fields
into ONE single, fluent bullet-point sentence per record, following the
STAR framework (Situation-Task-Action-Result) implicitly, AND tailor the
wording to align with the target job description's terminology and
emphasis.

Strict rules (must follow):
1. Do not add any concrete fact, number, scope, or degree of achievement
   that is not already present in the four source fields you are given.
2. Do not change the direction or strength of any claim -- if the source
   says "supported", do not upgrade it to "led"; if a result is described
   modestly, do not amplify it.
3. Every sentence must use an active voice and start with a strong action
   verb (e.g. "Redesigned...", "Implemented...", "Streamlined..."), not a
   passive construction (e.g. avoid "was tasked with", "was responsible
   for").
4. You may tailor WORD CHOICE to better match the job description's
   terminology and emphasis -- but you may not restructure the sentence's
   subject/responsibility (see rule 2) or introduce claims the source
   material doesn't support.
5. Do NOT touch, reformat, or remove any metric/number mentioned in the
   result field -- leave metric text exactly as given; a separate process
   formats numbers afterward.
6. You must obey any per-record operating notes given (notes_for_llm) as a
   hard constraint on top of the rules above.
7. Keep each bullet to one sentence, concise, resume-appropriate register
   -- no filler words, no first-person pronouns ("I", "my").
8. If any of the four source fields is empty or not provided, write the
   sentence using only the fields that ARE provided -- do not invent a
   substitute for a missing field.
9. Output must be in English, regardless of the input language.

Output format: one fused, JD-tailored bullet sentence per record, in the
same order as given, separated by the delimiter "|||" (three pipe
characters). No numbering, no preamble, no explanation -- output ONLY the
sentences separated by |||."""

REWRITE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Experience records to fuse and tailor ({count} total)
{records_block}

Output {count} fused, JD-tailored STAR bullet sentences, separated by |||:"""


def _fetch_governance_notes(session, claim_ids: list[str]) -> dict[str, str]:
    """claim_id -> notes_for_llm 映射，用於改寫時的硬性約束。"""
    if not claim_ids:
        return {}
    import uuid
    uuid_ids = [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]
    rows = session.query(GovernanceORM).filter(GovernanceORM.claim_id.in_(uuid_ids)).all()
    return {str(g.claim_id): (g.notes_for_llm or "none") for g in rows}


def _build_batch_prompt(
    claim_ids: list[str],
    contexts: dict[str, dict],
    notes_map: dict[str, str],
    job_description: str,
) -> tuple[str, list[str]]:
    """組出單一批次 prompt，把所有候選經歷的四段來源文字列出。"""
    ordered_claim_ids: list[str] = []
    lines = []
    for i, claim_id in enumerate(claim_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        ordered_claim_ids.append(claim_id)
        notes_for_llm = notes_map.get(claim_id, "none")
        lines.append(
            f"[Record {i}]\n"
            f"  Background: {ctx.get('project_context') or '(not provided)'}\n"
            f"  Problem: {ctx.get('problem_description') or '(not provided)'}\n"
            f"  Action: {ctx.get('action_description') or '(not provided)'}\n"
            f"  Result: {ctx.get('claim_description') or '(not provided)'}\n"
            f"  Operating notes: {notes_for_llm}"
        )

    records_block = "\n\n".join(lines)
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
    """單次批次 LLM 呼叫。回傳 {claim_id: 融合且已貼近JD的句子}。"""
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
    LLM 呼叫，逐條要求人工確認，回傳最終要用的完整句子文字。"""
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
