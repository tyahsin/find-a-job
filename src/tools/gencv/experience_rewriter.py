"""
experience_rewriter.py
Version: v6 (2026-08-19)

Changelog:
- v6 (2026-08-19): Migrated the REGENERATE gate in _confirm_one() to the
  shared core.llm_reviewer.parse_review_input() parser, replacing the
  two-step "type 'r', then answer a follow-up steering-note prompt" flow
  from v5 with a single-line command: 'r' (no steering note) or
  'r <text>' (steering note inline, in one input). This removes the extra
  round-trip and matches the same input contract now shared across
  claim_ranker.py, skill_grouper.py, and summary_writer.py. The
  fully-manual-override path is now spelled 'o <text>' (previously: any
  arbitrary typed text was implicitly treated as an override). Plain 'o'
  with no text, or any unrecognized input, is now explicitly invalid --
  the reviewer re-prompts with an error message instead of silently
  guessing what the user meant. Regeneration semantics (single-record LLM
  call, unlimited retries, steering note layered on top of the existing
  safety rules) are UNCHANGED from v5.
- v5 (2026-08-19): Added a REGENERATE option to _confirm_one()'s per-bullet
  confirmation gate -- see prior changelog entries for full rationale
  (two-step prompt flow, since replaced in v6 above).
- v4 (2026-08-16): Added PROJECT-GROUPED batch awareness -- claim_ids are
  grouped by project_id before building the batch prompt, and any project
  with more than one claim in the batch gets an explicit instruction to
  foreground a DIFFERENT capability angle per record, based only on what
  the source material actually supports.
- v3 (2026-08-16): Rewrote REWRITE_SYSTEM_PROMPT to fix bullets that were
  too long and unfocused: ~20-25 word length guideline, action-verb-first
  structure, Situation/Problem context omitted by default.
- v2 (2026-08-15): Merged STAR-fusion with JD-tailoring into ONE batch LLM
  call, followed by ONE mandatory human confirmation gate per bullet.
- v1 (2026-08-10, inferred): Initial two-pass STAR fusion + JD tailoring.

Safety design principles (unchanged since v2):
1. Metric numbers are NEVER touched by the LLM here.
2. governance.notes_for_llm is respected as a hard constraint.
3. The LLM may only reorganize/condense the four source fields it is
   given -- no new facts, numbers, scope, or degree of achievement.
4. Every bullet requires human confirmation (accept / regenerate with an
   optional steering note / manual override) -- no silent batch
   acceptance.
5. Capability-angle diversification is presentation-only -- it may
   reframe HOW an already-true fact is described, never invent a
   capability or achievement the source material doesn't support.
6. A regeneration steering note may only request a different
   presentation of the SAME source material -- it is passed to the LLM
   as an additional constraint layered on top of rules 1/2, never a way
   to introduce a new fact/number/scope. The human confirmation gate
   still has the final say on every regenerated draft, exactly as it
   does on the first draft.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from collections import defaultdict

from core.db import get_session, GovernanceORM
from core.llm_client import get_llm_client
from core.llm_reviewer import parse_review_input
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
   period-count rule. Cut adjectives, process detail, and any
   information that doesn't help a reader judge WHAT was done and HOW
   WELL, before cutting anything that affects the actual claim.
4. STRUCTURE: start the bullet with a strong action verb (e.g.
   "Redesigned...", "Implemented...", "Streamlined..."), not a passive
   construction and not a Situation/Problem clause. Action and Result
   are the primary content of the bullet.
5. The Situation/Problem context is OMITTED by default. Only include it
   when the problem itself signals meaningful scope, scale, or difficulty
   that a reader would otherwise miss. When included, keep it to a short
   clause at the START of the bullet -- never at the end.
6. CAPABILITY DIVERSITY WITHIN A PROJECT GROUP: when a project group
   contains MORE THAN ONE record, make sure each bullet in that group
   foregrounds a DIFFERENT capability angle, based on what its own
   source material actually emphasizes -- never invent or exaggerate a
   capability distinction that isn't genuinely supported. If the records
   in a group are genuinely about the same capability, do NOT force
   artificial variation.
7. You may tailor WORD CHOICE to better match the job description's
   terminology and emphasis -- but you may not restructure the sentence's
   subject/responsibility or introduce claims the source material
   doesn't support.
8. Do NOT touch, reformat, or remove any metric/number mentioned in the
   result field.
9. You must obey any per-record operating notes given (notes_for_llm) as
   a hard constraint on top of the rules above.
10. Concise, resume-appropriate register -- no filler words, no
    first-person pronouns.
11. If any of the four source fields is empty or not provided, write the
    sentence using only the fields that ARE provided.
12. Output must be in English, regardless of the input language.

Output format: one fused, JD-tailored bullet sentence per record, in the
same order as given, separated by the delimiter "|||" (three pipe
characters). No numbering, no preamble, no explanation."""

REWRITE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Experience records to fuse and tailor ({count} total, organized by project group)
{records_block}

Output {count} fused, JD-tailored STAR bullet sentences, in the same order as given above, separated by |||:"""

REGENERATE_SYSTEM_PROMPT = """You are a resume bullet-point writer. You previously fused a piece of
source material (project background, problem, action, result) into one
bullet-point sentence tailored to a job description. The user was not
fully satisfied with your previous draft and has given you a short
steering note describing what should be different this time.

Strict rules (must follow) -- IDENTICAL to your original task, still in
force:
1. Do not add any concrete fact, number, scope, or degree of achievement
   that is not already present in the four source fields you are given.
2. Do not change the direction or strength of any claim.
3. The user's steering note may ONLY request a different PRESENTATION of
   the same underlying facts -- tone, length, emphasis, word choice,
   which capability angle to foreground. It may NOT be used to introduce
   a new fact, number, or degree of achievement beyond the source
   material, even if the steering note seems to ask for that -- if the
   note conflicts with rules 1/2, follow rules 1/2 instead.
4. LENGTH: keep the bullet to approximately 20-25 words, one fluent
   sentence, starting with a strong action verb.
5. Do NOT touch, reformat, or remove any metric/number mentioned in the
   result field.
6. You must obey any operating notes given (notes_for_llm) as a hard
   constraint on top of the rules above.
7. Output must be in English, regardless of the input language.

Output format: output ONLY the single revised bullet sentence, no
preamble, no explanation, no quotation marks."""

REGENERATE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Source material for this one record
Background: {project_context}
Problem: {problem_description}
Action: {action_description}
Result: {claim_description}
Operating notes: {notes_for_llm}

# Your previous draft (the user was not satisfied with this)
{previous_draft}

# User's steering note for this regeneration
{steering_note}

Output the revised bullet sentence:"""


def _fetch_governance_notes(session, claim_ids):
    if not claim_ids:
        return {}
    import uuid
    uuid_ids = [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]
    rows = session.query(GovernanceORM).filter(GovernanceORM.claim_id.in_(uuid_ids)).all()
    return {str(g.claim_id): (g.notes_for_llm or "none") for g in rows}


def _group_claims_by_project(claim_ids, contexts):
    order = []
    groups = defaultdict(list)
    labels = {}
    for claim_id in claim_ids:
        ctx = contexts.get(claim_id)
        project_id = ctx.get("project_id") if ctx else None
        key = project_id if project_id else f"__no_project__{claim_id}"
        if key not in groups:
            order.append(key)
            labels[key] = (ctx.get("project_name") if ctx else None) or "Unlabeled project"
        groups[key].append(claim_id)
    return [(labels[key], groups[key]) for key in order]


def _build_batch_prompt(claim_ids, contexts, notes_map, job_description):
    project_groups = _group_claims_by_project(claim_ids, contexts)
    ordered_claim_ids = []
    record_number = 0
    blocks = []

    for project_label, group_claim_ids in project_groups:
        group_lines = []
        group_record_numbers = []
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


def _parse_batch_response(raw_response, expected_count):
    parts = [p.strip() for p in raw_response.split("|||")]
    if len(parts) < expected_count:
        parts += [""] * (expected_count - len(parts))
    return parts[:expected_count]


def _draft_fused_bullets(claim_ids, contexts, notes_map, job_description):
    if not claim_ids:
        return {}
    user_prompt, ordered_claim_ids = _build_batch_prompt(claim_ids, contexts, notes_map, job_description)
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


def _regenerate_one_bullet(ctx, notes_for_llm, job_description, previous_draft, steering_note):
    user_prompt = REGENERATE_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        project_context=ctx.get("project_context") or "(not provided)",
        problem_description=ctx.get("problem_description") or "(not provided)",
        action_description=ctx.get("action_description") or "(not provided)",
        claim_description=ctx.get("claim_description") or "(not provided)",
        notes_for_llm=notes_for_llm,
        previous_draft=previous_draft or "(no usable previous draft)",
        steering_note=steering_note or "(no specific change requested -- try an alternative phrasing)",
    )
    response = get_llm_client().generate_text(
        system_prompt=REGENERATE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=200,
    )
    return response.text.strip()


def _confirm_one(ctx, notes_for_llm, job_description, index, total, org_label, title_label, original, drafted):
    """(v6) Uses core.llm_reviewer.parse_review_input() for the confirmation
    gate. Commands: Enter=accept, 'r'/'r <text>'=regenerate, 'o <text>'=
    manual override. Plain 'o' or unrecognized input is invalid and
    re-prompts with an error message."""
    print(f"\n[{index}/{total}] [{org_label} - {title_label}]")

    if not drafted.strip():
        print("  [Notice] LLM did not produce a usable fused sentence for this record.")
        print(f"  Falling back to original result text: {original}")
        while True:
            raw = input("  Accept the fallback? (Enter to accept / 'o <text>' to override): ").strip()
            decision = parse_review_input(raw)
            if decision.action == "accept":
                return original
            if decision.action == "override":
                return decision.value
            print(f"  [Error] {decision.value}")

    current_draft = drafted

    while True:
        print(f"  Original result only: {original}")
        print(f"  Fused & JD-tailored bullet: {current_draft}")

        if original.strip() == current_draft.strip():
            print("  (Fused result is identical to the original -- using as-is)")
            return original

        raw = input(
            "  Accept this bullet? (Enter to accept / 'r' or 'r <text>' to regenerate / "
            "'o <text>' to override): "
        ).strip()
        decision = parse_review_input(raw)

        if decision.action == "regenerate":
            print("  Regenerating...")
            current_draft = _regenerate_one_bullet(
                ctx, notes_for_llm, job_description, current_draft, decision.value
            )
            continue

        if decision.action == "override":
            print("  -> Using manually entered text")
            return decision.value

        if decision.action == "accept":
            print("  -> Using fused & tailored bullet")
            return current_draft

        print(f"  [Error] {decision.value}")


def rewrite_and_confirm(claim_ids, job_description):
    with get_session() as session:
        contexts = fetch_experience_contexts(session, claim_ids)
        notes_map = _fetch_governance_notes(session, claim_ids)

    print(f"\n=== Fusing background/problem/action/result + tailoring to JD for {len(claim_ids)} record(s) ===")

    drafts = _draft_fused_bullets(claim_ids, contexts, notes_map, job_description)

    final_descriptions = {}
    for i, claim_id in enumerate(claim_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        org_label = ctx.get("org_name") or "No organization"
        title_label = ctx.get("title") or ""
        original = ctx.get("claim_description") or ""
        drafted = drafts.get(claim_id, "")
        notes_for_llm = notes_map.get(claim_id, "none")

        final_descriptions[claim_id] = _confirm_one(
            ctx, notes_for_llm, job_description,
            i, len(claim_ids), org_label, title_label, original, drafted
        )

    print("=" * 50)
    return final_descriptions
