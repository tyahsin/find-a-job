"""
gencv/claim_rewriter.py

Safety design principles:
1. Metric numbers are NEVER touched by the LLM here.
2. governance.notes_for_llm is respected as a hard constraint.
3. The LLM may only reorganize/condense the source fields it is given for
   a record (one claim for a "single" record, 2+ claims for a "merge"
   record) -- no new facts, numbers, scope, or degree of achievement, and
   no facts borrowed from claims outside that one record.
4. Every bullet requires human confirmation (accept / regenerate with an
   optional steering note / manual override) -- no silent batch
   acceptance.
5. Capability-angle diversification is presentation-only -- it may
   reframe HOW an already-true fact is described, never invent a
   capability or achievement the source material doesn't support. The
   angle itself now comes from claim_ranker's own KEEP/MERGE labels
   (item["reasons"]) rather than being guessed here.
6. A regeneration steering note may only request a different
   presentation of the SAME source material -- it is passed to the LLM
   as an additional constraint layered on top of rules 1/2, never a way
   to introduce a new fact/number/scope. The human confirmation gate
   still has the final say on every regenerated draft, exactly as it
   does on the first draft.
7. A "merge" record's sub-records may be fused (combining complementary
   details) but never expanded into more than one sentence, and never
   allowed to silently drop a sub-record's content -- if two sub-records
   conflict or overlap entirely, prefer the more specific/complete one
   rather than inventing a synthesis that isn't supported by either.
8. action.role_scope (led / collaborated / supported), when present, is
   respected as a hard constraint the same way notes_for_llm is -- the
   bullet's framing must match it, never inflate or deflate it.

No DB access here (by design): claim_selector.py's single SQL query already
pulled every field this whole claim-progression pipeline needs -- including
notes_for_llm and org_name -- onto each claim's context dict. claim_ranker.py
carries that SAME contexts dict through narrow_all_claims and hands it back
to the caller; this module just receives it as a parameter and reads
whatever fields it needs off of it. If a future change reintroduces a
need for a field claim_selector.py doesn't already select, add it to that
one SQL query -- don't add a second DB round trip here.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

import json
import re
from collections import defaultdict

from core.llm_client import get_llm_client
from core.llm_reviewer import parse_review_input

REWRITE_SYSTEM_PROMPT = """You are a resume bullet-point writer. You will be given several RECORDS to
turn into resume bullets, plus the target job description. Each record is
either a SINGLE source (one project background/problem/action/result) or a
MERGE of 2+ sources that have ALREADY been judged to describe the SAME
underlying accomplishment -- for a merge record, synthesize all of its
sub-records into ONE fused bullet, never one sentence per sub-record.
Records are organized into GROUPS; records in the same group came from the
same position or project. Some records carry a "Capability angle" label --
this is a hint for what the bullet should foreground, decided upstream; use
it to guide phrasing, never as license to state something the source
material doesn't support.

Strict rules (must follow):
1. Do not add any concrete fact, number, scope, or degree of achievement
   that is not already present in the record's own source field(s). Never
   borrow a fact from a different record, even one in the same group.
2. Do not change the direction or strength of any claim -- if the source
   says "supported", do not upgrade it to "led"; if a result is described
   modestly, do not amplify it.
3. ROLE SCOPE: when a record gives a "Role scope" (led / collaborated /
   supported), the bullet's verb and framing must match it exactly -- a
   "supported" record must never read as if the candidate led or drove it
   independently, and a "led" record must never be softened into passive
   team language. This is a hard constraint, not a stylistic suggestion,
   and applies per sub-record in a merge (do not let one sub-record's
   stronger role scope bleed into how another sub-record's contribution is
   described).
4. LENGTH: keep each bullet to approximately 20-25 words -- roughly one
   to two lines at typical resume column width. This is a WORD-COUNT
   guideline based on rendered line length, NOT a sentence-count or
   period-count rule. Cut adjectives, process detail, and any
   information that doesn't help a reader judge WHAT was done and HOW
   WELL, before cutting anything that affects the actual claim.
5. STRUCTURE: start the bullet with a strong action verb (e.g.
   "Redesigned...", "Implemented...", "Streamlined..."), not a passive
   construction and not a Situation/Problem clause. Action and Result
   are the primary content of the bullet.
6. The Situation/Problem context is OMITTED by default. Only include it
   when the problem itself signals meaningful scope, scale, or difficulty
   that a reader would otherwise miss. When included, keep it to a short
   clause at the START of the bullet -- never at the end.
7. CAPABILITY DIVERSITY WITHIN A GROUP: when a group contains MORE THAN
   ONE record, make sure each bullet foregrounds a DIFFERENT capability,
   consistent with each record's own "Capability angle" label when given.
   Never invent or exaggerate a distinction the source material doesn't
   support; if two records in a group are genuinely about the same
   capability despite different labels, do not force artificial variation.
8. MERGE RECORDS: when a record merges 2+ sub-records, combine their
   complementary details into ONE sentence -- never repeat the same fact
   twice, never produce more than one sentence, and never drop a
   sub-record's content just because it didn't fit; prefer trimming
   phrasing over dropping a distinct fact.
9. You may tailor WORD CHOICE to better match the job description's
   terminology and emphasis -- but you may not restructure the sentence's
   subject/responsibility or introduce claims the source material
   doesn't support.
10. Do NOT touch, reformat, or remove any metric/number mentioned in any
    Result field.
11. You must obey any per-record operating notes given (notes_for_llm) as
    a hard constraint on top of the rules above -- when a merge record has
    multiple operating notes, obey all of them.
12. Concise, resume-appropriate register -- no filler words, no
    first-person pronouns.
13. If a source field is empty or not provided, write the sentence using
    only the fields that ARE provided.
14. Output must be in English, regardless of the input language.

Output format: output ONLY a valid JSON object mapping each record's number
(as a STRING key, e.g. "1", "2") to its fused, JD-tailored bullet sentence
-- one entry per RECORD (not per sub-record), covering EVERY record number
shown to you, exactly once each. No preamble, no explanation, no markdown
code fences.
Example shape (illustrative only, not real content): {{"1": "Redesigned the onboarding flow, reducing drop-off by 18%.", "2": "Implemented an automated testing pipeline, cutting deploy time by 40%."}}"""

REWRITE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Records to fuse and tailor ({count} total, organized by group)
{records_block}

Output a JSON object with exactly {count} entries, one fused, JD-tailored bullet sentence per record number ("1" through "{count}"):"""

REGENERATE_SYSTEM_PROMPT = """You are a resume bullet-point writer. You previously fused a record's source
material into one bullet-point sentence tailored to a job description. The
user was not fully satisfied with your previous draft and has given you a
short steering note describing what should be different this time.

Strict rules (must follow) -- IDENTICAL to your original task, still in
force:
1. Do not add any concrete fact, number, scope, or degree of achievement
   that is not already present in the record's own source field(s).
2. Do not change the direction or strength of any claim.
3. ROLE SCOPE: when a record gives a "Role scope" (led / collaborated /
   supported), the bullet's verb and framing must match it exactly, per
   sub-record if this record is a merge -- this is a hard constraint, not
   a stylistic suggestion.
4. If this record merges 2+ sub-records, keep fusing all of them into ONE
   sentence -- never revert to one sentence per sub-record, never drop a
   sub-record's content.
5. The user's steering note may ONLY request a different PRESENTATION of
   the same underlying facts -- tone, length, emphasis, word choice,
   which capability angle to foreground. It may NOT be used to introduce
   a new fact, number, or degree of achievement beyond the source
   material, even if the steering note seems to ask for that -- if the
   note conflicts with rules 1/2/3/4, follow rules 1/2/3/4 instead.
6. LENGTH: keep the bullet to approximately 20-25 words, one fluent
   sentence, starting with a strong action verb.
7. Do NOT touch, reformat, or remove any metric/number mentioned in any
   Result field.
8. You must obey any operating notes given (notes_for_llm) as a hard
   constraint on top of the rules above.
9. Output must be in English, regardless of the input language.

Output format: output ONLY the single revised bullet sentence, no
preamble, no explanation, no quotation marks."""

REGENERATE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Source material for this record
{source_block}

# Your previous draft (the user was not satisfied with this)
{previous_draft}

# User's steering note for this regeneration
{steering_note}

Output the revised bullet sentence:"""


def _item_key(item):
    """Identifies one claim_ranker item for matching drafts back up in
    rewrite_and_confirm -- (group_key, claim_ids) is unique because
    claim_ranker never lets the same claim_id appear in two items within one
    group, and different groups always carry different group_key values."""
    return (item.get("group_key"), tuple(item["claim_ids"]))


def _group_items_by_label(items):
    """Groups items by the group_key/group_label claim_ranker already
    assigned them (one position or one project) -- preserving first-seen
    order. This is presentation-only (deciding when to print a
    capability-diversity note); the actual MECE narrowing already happened
    upstream in claim_ranker.py."""
    order = []
    groups = defaultdict(list)
    labels = {}
    for item in items:
        key = item.get("group_key")
        if key not in groups:
            order.append(key)
            labels[key] = item.get("group_label") or "Unlabeled group"
        groups[key].append(item)
    return [(labels[key], groups[key]) for key in order]


def _build_item_source_block(item, contexts):
    """Builds the source-material block for ONE record ('single' or
    'merge'), shared between the first batch draft and any later
    regeneration of just this record. A 'merge' record shows every member
    claim as its own labeled sub-record so the LLM can see exactly what it
    is and isn't allowed to draw from. contexts is claim_ranker's own
    claim_id -> full-row dict (already carries notes_for_llm) -- no
    separate governance lookup needed."""
    claim_ids = [cid for cid in item["claim_ids"] if cid in contexts]
    reasons = item.get("reasons") or {}
    labels = sorted({reasons[cid] for cid in claim_ids if reasons.get(cid)})
    label_line = f"    Capability angle: {' / '.join(labels)}\n" if labels else ""

    def _fields(cid):
        ctx = contexts[cid]
        notes_for_llm = ctx.get("notes_for_llm") or "none"
        role_scope = ctx.get("role_scope")
        role_line = (
            f"    Role scope: {role_scope} (hard constraint -- do not describe this "
            f"as more or less independent than stated, see rule 3)\n"
            if role_scope else ""
        )
        return (
            f"    Background: {ctx.get('project_context') or '(not provided)'}\n"
            f"    Problem: {ctx.get('problem_description') or '(not provided)'}\n"
            f"    Action: {ctx.get('action_description') or '(not provided)'}\n"
            f"{role_line}"
            f"    Result: {ctx.get('claim_description') or '(not provided)'}\n"
            f"    Operating notes: {notes_for_llm}"
        )

    if item.get("kind") != "merge" or len(claim_ids) == 1:
        return label_line + _fields(claim_ids[0])

    sub_blocks = [f"    -- Sub-record {i} --\n{_fields(cid)}" for i, cid in enumerate(claim_ids, 1)]
    merge_note = (
        f"    NOTE: this record MERGES {len(claim_ids)} sub-records already judged to describe the "
        f"SAME underlying accomplishment -- fuse them into ONE bullet (rule 7), never one sentence "
        f"per sub-record.\n"
    )
    return label_line + merge_note + "\n".join(sub_blocks)


def _build_batch_prompt(items, contexts, job_description):
    grouped = _group_items_by_label(items)
    ordered_items = []
    record_number = 0
    blocks = []

    for group_label, group_items in grouped:
        group_lines = []
        group_record_numbers = []
        for item in group_items:
            if not any(cid in contexts for cid in item["claim_ids"]):
                continue
            record_number += 1
            ordered_items.append(item)
            group_record_numbers.append(record_number)
            kind_tag = "merge" if item.get("kind") == "merge" else "single"
            group_lines.append(
                f"  [Record {record_number}] (kind: {kind_tag})\n"
                + _build_item_source_block(item, contexts)
            )
        if not group_lines:
            continue

        group_header = f"# Group: {group_label} ({len(group_record_numbers)} record(s))"
        if len(group_record_numbers) > 1:
            group_header += (
                f"\n  NOTE: Records {', '.join(str(n) for n in group_record_numbers)} are in the "
                f"SAME group -- apply rule 6 (capability diversity within a group), using each "
                f"record's own Capability angle label where given."
            )
        blocks.append(group_header + "\n" + "\n\n".join(group_lines))

    records_block = "\n\n".join(blocks)
    user_prompt = REWRITE_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(ordered_items),
        records_block=records_block,
    )
    return user_prompt, ordered_items


def _parse_batch_response(raw_response, expected_count):
    """Parses the LLM's JSON object ({"1": "...", "2": "...", ...}) into an
    ordered list of `expected_count` bullet strings (index i = record i+1).
    Unlike the old "|||".split() approach, a structural mismatch (missing
    record number, extra/unexpected key, unparseable JSON) is DETECTED and
    reported here rather than silently shifting every later record's
    bullet into the wrong slot -- any record that couldn't be recovered
    gets "" (the caller's existing empty-string fallback path in
    _confirm_item already handles that per-record, requiring the human to
    accept the raw-source fallback or manually override just that one
    record, instead of the whole batch being silently misaligned)."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(
            f"  [Notice] Could not parse the LLM's batch response as JSON ({exc}) -- "
            f"every record in this batch will fall back to its raw source text; "
            f"you can 'r'/'o' individual records afterward."
        )
        return [""] * expected_count

    if not isinstance(parsed, dict):
        print(
            "  [Notice] LLM's batch response was valid JSON but not an object "
            "(expected {\"1\": \"...\", ...}) -- every record in this batch will "
            "fall back to its raw source text."
        )
        return [""] * expected_count

    expected_keys = {str(i) for i in range(1, expected_count + 1)}
    got_keys = set(parsed.keys())
    missing = expected_keys - got_keys
    extra = got_keys - expected_keys
    if missing or extra:
        print(
            f"  [Notice] Batch response key mismatch -- missing: {sorted(missing) or 'none'}, "
            f"unexpected: {sorted(extra) or 'none'}. Missing records fall back to raw source "
            f"text; unexpected keys are ignored."
        )

    return [str(parsed.get(str(i), "")).strip() for i in range(1, expected_count + 1)]


def _draft_fused_bullets(items, contexts, job_description):
    print(items)
    if not items:
        return {}
    user_prompt, ordered_items = _build_batch_prompt(items, contexts, job_description)
    print(user_prompt)
    print(ordered_items)
    if not ordered_items:
        return {}
    response = get_llm_client().generate_text(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=200 * len(ordered_items),
    )
    sentences = _parse_batch_response(response.text, len(ordered_items))
    return {_item_key(item): sentence for item, sentence in zip(ordered_items, sentences)}


def _regenerate_one_item(item, contexts, job_description, previous_draft, steering_note):
    source_block = _build_item_source_block(item, contexts)
    user_prompt = REGENERATE_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        source_block=source_block,
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


def _confirm_item(item, contexts, job_description, index, total, drafted):
    """Confirms one claim_ranker item -- 'single' or 'merge' -- via
    core.llm_reviewer.parse_review_input(). Commands: Enter=accept, 'r'/'r
    <text>'=regenerate, 'o <text>'=manual override. Plain 'o' or
    unrecognized input is invalid and re-prompts with an error message."""
    claim_ids = [cid for cid in item["claim_ids"] if cid in contexts]
    group_label = item.get("group_label") or "Unlabeled group"
    org_label = (contexts.get(claim_ids[0], {}).get("org_name") if claim_ids else None) or "No organization"
    kind_tag = "merged" if item.get("kind") == "merge" else "single"
    print(f"\n[{index}/{total}] [{org_label} - {group_label}] ({kind_tag}, {len(claim_ids)} claim(s))")

    # For a merge record this is a raw, unfused concatenation -- only ever
    # used as a fallback when the LLM produced nothing usable, or as the
    # "identical to source" shortcut when the LLM echoed it back verbatim.
    original = "; ".join(
        contexts[cid]["claim_description"] for cid in claim_ids if contexts[cid].get("claim_description")
    )

    if not drafted.strip():
        print("  [Notice] LLM did not produce a usable fused sentence for this record.")
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
            current_draft = _regenerate_one_item(
                item, contexts, job_description, current_draft, decision.value
            )
            continue

        if decision.action == "override":
            print("  -> Using manually entered text")
            return decision.value

        if decision.action == "accept":
            print("  -> Using fused & tailored bullet")
            return current_draft

        print(f"  [Error] {decision.value}")


def rewrite_and_confirm(items, contexts, job_description):
    """items: the unified {kind, claim_ids, group_key, group_label, track,
    reasons} list produced by gencv/claim_ranker.py (narrow_all_claims, or
    narrow_position_group/select_top_projects directly). contexts: the SAME
    claim_id -> full-row dict narrow_all_claims returns alongside items --
    already carries every field this module needs (project_context,
    problem_description, action_description, claim_description,
    notes_for_llm, org_name), because claim_selector.py's one SQL query
    fetched them all up front. No DB access happens in this module.

    A "single" item becomes one bullet from one claim's source material; a
    "merge" item fuses ALL of its member claims' source material into ONE
    bullet, since claim_ranker already judged them to describe the same
    underlying accomplishment -- this module no longer re-derives grouping
    or diversity on its own, it consumes claim_ranker's decisions directly.

    Returns a list of dicts: each input item plus a "description" key
    holding the final, human-confirmed bullet text -- group_key,
    group_label, track, kind, and reasons all survive unchanged for
    whatever assembles the resume section next."""
    print(f"\n=== Fusing source material + tailoring to JD for {len(items)} record(s) ===")

    drafts = _draft_fused_bullets(items, contexts, job_description)

    finalized = []
    for i, item in enumerate(items, 1):
        if not any(cid in contexts for cid in item["claim_ids"]):
            continue
        drafted = drafts.get(_item_key(item), "")
        description = _confirm_item(item, contexts, job_description, i, len(items), drafted)
        finalized.append({**item, "description": description})

    print("=" * 50)
    return finalized