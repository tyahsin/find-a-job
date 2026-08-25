"""
claim_ranker.py
Version: v3 (2026-08-19)

Changelog:
- v3 (2026-08-19): Migrated the REGENERATE gate in
  _confirm_project_selection() to the shared core.llm_reviewer.
  parse_review_input() parser -- same single-line command contract as
  experience_rewriter.py v6 / skill_grouper.py v3 / summary_writer.py v3:
  'r' or 'r <text>' to regenerate (previously a two-step prompt: type
  'r', then answer a separate steering-note question). The manual
  override path is now spelled 'o <comma-separated numbers>' instead of
  bare numbers -- keeps the input contract identical across all four
  reviewers while still supporting this module's KEEP+DROP index-based
  override. Plain 'o' with no numbers, or any unrecognized input, is now
  explicitly invalid and re-prompts with an error. Regeneration semantics
  (forcing the LLM fallback judgment on a project whose primary result
  came from the deterministic tag algorithm, optionally guided by a
  steering note, logged to CLAIM_RANKER_REGEN_LOG) are UNCHANGED from v2.
- v2 (2026-08-19): Added a REGENERATE option to
  _confirm_project_selection()'s lightweight confirmation gate, plus an
  observation log for tracking how often it gets used -- see prior
  changelog entry for full rationale (two-step prompt flow, since
  replaced in v3 above).
- v1 (2026-08-16): Initial version. Solves the "same-project claim
  redundancy" problem via tag-Jaccard-based capability-diversity
  selection, with an LLM fallback when tag coverage is insufficient.

MAX_CLAIMS_PER_PROJECT = 2 (hard cap). Primary signal: Jaccard similarity
over each claim's tag_name set -- deterministic, zero LLM calls in the
common case. LLM fallback triggers on insufficient tag coverage OR
(v2+) explicit user regenerate request.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from core.db import get_session
from core.llm_client import get_llm_client
from core.llm_reviewer import parse_review_input
from gencv.experience_matcher import fetch_experience_contexts, fetch_tags_for_claims

MAX_CLAIMS_PER_PROJECT = 2

# CLAIM_RANKER_REGEN_LOG = Path("claim_ranker_regeneration_log.jsonl")

RANKER_SYSTEM_PROMPT = """You are a resume content curator. You will be given a job description and
several candidate experience records that ALL belong to the SAME project
and have ALREADY been judged relevant to this job -- your task is NOT to
re-judge relevance, it is to pick which ones to keep when there are more
candidates than the resume has room for.

Your task: identify the core CAPABILITY each record demonstrates, then
select exactly {max_claims} record(s) whose capabilities are as DIFFERENT
from each other as possible, so the resume shows the candidate's range
rather than repeating the same strength.

{steering_note_block}

Strict rules (must follow):
1. Every record given to you has already passed a relevance check -- do
   not exclude a record for being "less relevant"; your only basis for
   choosing is capability diversity (and the user's steering note, if
   given, as an additional preference on TOP OF capability diversity --
   never as a reason to select a record that wasn't already judged
   relevant).
2. If the candidate records are GENUINELY single-faceted, say so
   explicitly and select the {max_claims} record(s) that best represent
   the project's overall contribution instead. Do NOT invent or
   exaggerate a capability distinction that isn't actually supported.
3. Do not add any fact, number, or claim not already present in the
   records given.
4. Output must be in English, regardless of input language.

Output format: first line is either "DIVERSE" or "SINGLE-FACETED". Second
line is the selected record number(s), comma-separated (exactly
{max_claims}, or fewer only if fewer than {max_claims} records were
given). No other text."""

RANKER_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Candidate records from the same project ({count} total, all already JD-relevant)
{records_block}

Output your assessment (DIVERSE or SINGLE-FACETED) and the {max_claims} selected record number(s):"""


def _group_claims_by_project(claim_ids, contexts):
    grouped = defaultdict(list)
    for claim_id in claim_ids:
        ctx = contexts.get(claim_id)
        project_id = ctx.get("project_id") if ctx else None
        key = project_id if project_id else f"__no_project__{claim_id}"
        grouped[key].append(claim_id)
    return grouped


def _jaccard_similarity(tags_a, tags_b):
    if not tags_a and not tags_b:
        return 0.0
    union = tags_a | tags_b
    if not union:
        return 0.0
    return len(tags_a & tags_b) / len(union)


def _select_by_tag_diversity(candidate_ids, tags_map, max_claims):
    tag_sets = {cid: set(tags_map.get(cid, [])) for cid in candidate_ids}
    if len(candidate_ids) <= max_claims:
        return list(candidate_ids)

    pairwise = {}
    for a, b in combinations(candidate_ids, 2):
        sim = _jaccard_similarity(tag_sets[a], tag_sets[b])
        pairwise[(a, b)] = sim
        pairwise[(b, a)] = sim

    def avg_similarity_to(cid, others):
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


def _has_full_tag_coverage(candidate_ids, tags_map):
    return all(tags_map.get(cid) for cid in candidate_ids)


def _build_ranker_prompt(candidate_ids, contexts, job_description, max_claims, steering_note=None):
    index_to_claim_id = {}
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

    if steering_note:
        steering_note_block = (
            "The user was not satisfied with a previous automated selection for this "
            "project and gave this steering note -- treat it as an additional preference "
            "on top of capability diversity (never as license to select a record that "
            "wasn't already judged relevant, or to invent facts not in the records):\n"
            f'"{steering_note}"'
        )
    else:
        steering_note_block = ""

    user_prompt = RANKER_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(index_to_claim_id),
        records_block=records_block,
        max_claims=max_claims,
    )
    return user_prompt, index_to_claim_id, steering_note_block


def _select_by_llm_judgment(candidate_ids, contexts, job_description, max_claims, steering_note=None):
    user_prompt, index_to_claim_id, steering_note_block = _build_ranker_prompt(
        candidate_ids, contexts, job_description, max_claims, steering_note
    )
    if not index_to_claim_id:
        return candidate_ids[:max_claims], False

    response = get_llm_client().generate_text(
        system_prompt=RANKER_SYSTEM_PROMPT.format(max_claims=max_claims, steering_note_block=steering_note_block),
        user_prompt=user_prompt,
        temperature=0.0 if not steering_note else 0.3,
        max_tokens=100,
    )
    lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
    is_single_faceted = bool(lines) and "SINGLE-FACETED" in lines[0].upper()

    indices = []
    if len(lines) >= 2:
        indices = [int(n) for n in re.findall(r"\d+", lines[1])]
    elif lines:
        indices = [int(n) for n in re.findall(r"\d+", lines[0])]

    selected = [index_to_claim_id[i] for i in indices if i in index_to_claim_id][:max_claims]
    if not selected:
        selected = candidate_ids[:max_claims]

    return selected, is_single_faceted


# def _log_regeneration_event(project_label, previous_method, steering_note):
#     entry = {
#         "timestamp": datetime.now(timezone.utc).isoformat(),
#         "project_label": project_label,
#         "previous_method": previous_method,
#         "steering_note": steering_note or "(none provided)",
#     }
#     try:
#         with CLAIM_RANKER_REGEN_LOG.open("a", encoding="utf-8") as f:
#             f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#     except OSError as exc:
#         print(f"  [Notice] Could not write regeneration observation log: {exc}")


def _confirm_project_selection(project_label, kept_ids, dropped_ids, contexts, method_note,
                                candidate_ids, job_description, max_claims):
    """(v3) Uses core.llm_reviewer.parse_review_input(). Commands:
    Enter=accept, 'r'/'r <text>'=regenerate (forces LLM fallback judgment),
    'o <comma-separated numbers>'=manual override using 1-based positions
    in KEEP+DROP order. Plain 'o' or unrecognized input is invalid and
    re-prompts."""
    if not dropped_ids:
        return kept_ids

    current_kept, current_dropped = kept_ids, dropped_ids
    current_method_note = method_note

    while True:
        print(f"\n=== Project \"{project_label}\": narrowed from {len(current_kept) + len(current_dropped)} to {len(current_kept)} claim(s) ({current_method_note}) ===")
        for cid in current_kept:
            desc = contexts.get(cid, {}).get("claim_description", "")
            print(f"  [KEEP] {desc}")
        for cid in current_dropped:
            desc = contexts.get(cid, {}).get("claim_description", "")
            print(f"  [DROP] {desc}")
        print("  (Dropped claims are excluded from THIS resume only -- they remain in your "
              "database and may be selected again for a different job description.)")

        raw = input(
            "  Accept this selection? (Enter to accept / 'r' or 'r <text>' to regenerate / "
            "'o <comma-separated numbers>' to override, using 1-based position in KEEP+DROP order): "
        ).strip()
        decision = parse_review_input(raw)

        if decision.action == "regenerate":
            steering_note = decision.value
            # _log_regeneration_event(project_label, current_method_note, steering_note or "")

            print("  Regenerating via LLM judgment...")
            new_kept, is_single_faceted = _select_by_llm_judgment(
                candidate_ids, contexts, job_description, max_claims, steering_note=steering_note
            )
            new_dropped = [cid for cid in candidate_ids if cid not in new_kept]
            current_kept, current_dropped = new_kept, new_dropped
            current_method_note = (
                "LLM judgment: single-faceted, selected by overall fit (regenerated)"
                if is_single_faceted
                else "LLM judgment: capability diversity (regenerated)"
            )
            continue

        if decision.action == "accept":
            return current_kept

        if decision.action == "override":
            all_ids = current_kept + current_dropped
            try:
                indices = [int(x.strip()) for x in decision.value.split(",") if x.strip()]
                override = [all_ids[i - 1] for i in indices]
                print(f"  -> Using manually selected: {len(override)} claim(s)")
                return override
            except (ValueError, IndexError):
                print("  [Error] Could not parse override numbers -- try again.")
                continue

        print(f"  [Error] {decision.value}")


def rank_and_narrow_claims(claim_ids, job_description):
    if not claim_ids:
        return []

    with get_session() as session:
        contexts = fetch_experience_contexts(session, claim_ids)
        tags_map = fetch_tags_for_claims(session, claim_ids)

    grouped = _group_claims_by_project(claim_ids, contexts)
    final_ids = []

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
        confirmed = _confirm_project_selection(
            project_label, kept, dropped, contexts, method_note,
            group_claim_ids, job_description, MAX_CLAIMS_PER_PROJECT,
        )
        final_ids.extend(confirmed)

    return final_ids
