"""
skill_grouper.py
Version: v3 (2026-08-19)

Changelog:
- v3 (2026-08-19): Migrated the REGENERATE gate in confirm_skill_grouping()
  to the shared core.llm_reviewer.parse_review_input() parser -- same
  single-line command contract as experience_rewriter.py v6 /
  claim_ranker.py v3 / summary_writer.py v3: 'r' or 'r <text>' to
  regenerate (previously a two-step prompt). Plain 'o' with no text, or
  any unrecognized input, is now explicitly invalid and re-prompts with
  an error. Regeneration semantics (wholesale replacement of the current
  draft grouping, discarding any manual edits made so far in this
  session, warning displayed before proceeding) are UNCHANGED from v2.
- v2 (2026-08-19): Added a REGENERATE option to confirm_skill_grouping()'s
  confirmation gate -- see prior changelog entry for full rationale
  (two-step prompt flow, since replaced in v3 above).
- v1 (2026-08-16): Initial version. Presentation-layer regrouping of
  already-confirmed tags into JD-appropriate resume skill-section groups
  (max 4 groups), with mandatory human confirmation, interactive
  unplaced-tag resolution (no automatic "Other" bucket), and an
  interactive add-new-group step.

Safety rules (unchanged): the LLM may ONLY re-bucket existing tag_name
strings into new group labels -- it may NOT invent, rename, merge the
text of, or drop any tag_name.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from __future__ import annotations

import json
import re

from core.llm_client import get_llm_client
from core.llm_reviewer import parse_review_input

MAX_GROUPS = 4

GROUPING_SYSTEM_PROMPT = """You are a resume skills-section organizer. You will be given a target job
description and a flat list of skill/knowledge tags the candidate has
already confirmed apply to this resume. Each tag also comes with its
original data-layer category (skill / domain / job_function) as a hint,
but that category system is NOT meant for resume presentation.

Your task: propose new group labels and assign every tag to exactly one
group, choosing groupings that make sense for THIS job description's
domain and terminology.

{steering_note_block}

Strict rules (must follow):
1. HARD LIMIT: produce AT MOST 4 groups. Fewer is fine if that better
   fits the tags given -- do not pad to reach 4.
2. Try to place every tag in the input list into exactly one output
   group. If a tag genuinely does not fit any sensible group, it is
   better to leave it out than to force it somewhere wrong.
3. Do NOT change the text of any tag -- copy each tag_name exactly as
   given, verbatim.
4. The original category is a hint, not a hard constraint -- you are
   free to split, merge, or relabel across that boundary if the job
   description calls for it.
5. Choose group labels that are natural, resume-appropriate section
   headers (Title Case, 1-4 words each).
6. Output must be in English, regardless of input language.

Output format: output ONLY a valid JSON object mapping each group label
to a list of tag_name strings, and nothing else -- no preamble, no
markdown code fences.
Example shape (illustrative only): {{"Programming Languages": ["Python", "SQL"], "Cloud & Infrastructure": ["Docker", "AWS"]}}"""

GROUPING_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Tags to group ({count} total)
{tags_block}

Output the grouped JSON (at most {max_groups} groups):"""


def _format_tags_block(tags_with_category):
    lines = []
    for tag_name, tag_category in tags_with_category:
        lines.append(f"- {tag_name} (original category: {tag_category})")
    return "\n".join(lines)


def _parse_grouping_response(raw_response, all_tag_names):
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    seen = set()
    result = {}
    for label, tags in list(parsed.items())[:MAX_GROUPS]:
        if not isinstance(tags, list):
            continue
        valid_tags = [t for t in tags if isinstance(t, str) and t in all_tag_names and t not in seen]
        if valid_tags:
            result[str(label)] = valid_tags
            seen.update(valid_tags)

    return result


def draft_skill_grouping(tags_with_category, job_description, steering_note=None):
    if not tags_with_category:
        return {}

    all_tag_names = {t for t, _ in tags_with_category}

    if steering_note:
        steering_note_block = (
            "The user was not satisfied with a previous grouping attempt and gave this "
            "steering note for how to group differently this time -- treat it as an "
            "additional preference on top of the rules above (never as license to rename, "
            "invent, or drop a tag_name):\n"
            f'"{steering_note}"'
        )
    else:
        steering_note_block = ""

    user_prompt = GROUPING_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(tags_with_category),
        tags_block=_format_tags_block(tags_with_category),
        max_groups=MAX_GROUPS,
    )
    response = get_llm_client().generate_text(
        system_prompt=GROUPING_SYSTEM_PROMPT.format(steering_note_block=steering_note_block),
        user_prompt=user_prompt,
        temperature=0.2 if not steering_note else 0.4,
        max_tokens=800,
    )
    return _parse_grouping_response(response.text, all_tag_names)


def _prompt_assign_unplaced_tag(tag_name, group_labels):
    print(f"\n  Unplaced tag: '{tag_name}'")
    for i, label in enumerate(group_labels, 1):
        print(f"    [{i}] {label}")
    print(f"    [{len(group_labels) + 1}] Create a new group for this tag")

    choice = input("  Assign to which group? (enter a number, or type a new group name): ").strip()

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(group_labels):
            return group_labels[idx - 1]
        if idx == len(group_labels) + 1:
            new_label = input("  New group name: ").strip()
            return new_label or "Skills"

    return choice if choice else "Skills"


def _resolve_unplaced_tags(grouping, all_tag_names):
    placed = {t for tags in grouping.values() for t in tags}
    unplaced = sorted(all_tag_names - placed)

    if not unplaced:
        return grouping

    print(f"\n=== {len(unplaced)} tag(s) were not placed into a group by the draft ===")
    for tag_name in unplaced:
        target_label = _prompt_assign_unplaced_tag(tag_name, list(grouping.keys()))
        grouping.setdefault(target_label, []).append(tag_name)

    return grouping


def _prompt_add_new_groups(grouping):
    choice = input("\n  Add a new group by moving tags out of existing ones? (y/N): ").strip().lower()

    if choice != "y":
        return grouping

    while True:
        new_label = input("  New group name (leave blank to stop adding groups): ").strip()
        if not new_label:
            break

        print("  Current tags by group:")
        for label, tags in grouping.items():
            print(f"    [{label}]: {', '.join(tags)}")

        raw_tags = input(f"  Which tags should move into '{new_label}'? (comma-separated, exact names): ").strip()
        tags_to_move = [t.strip() for t in raw_tags.split(",") if t.strip()]

        moved = []
        for label in list(grouping.keys()):
            remaining = []
            for t in grouping[label]:
                if t in tags_to_move and t not in moved:
                    moved.append(t)
                else:
                    remaining.append(t)
            grouping[label] = remaining

        grouping[new_label] = moved
        grouping = {label: tags for label, tags in grouping.items() if tags}

        print(f"  -> Created '{new_label}' with: {', '.join(moved) if moved else '(no tags matched)'}")

    return grouping


def confirm_skill_grouping(drafted_grouping, all_tag_names, tags_with_category=None, job_description=None):
    """(v3) Uses core.llm_reviewer.parse_review_input(). Commands:
    Enter=accept, 'r'/'r <text>'=regenerate. Any other input (including
    plain 'o') is invalid here -- this gate has no manual free-text
    override of its own; manual adjustments happen via the unplaced-tag
    and add-new-group steps below."""
    print("\n=== Proposed skills-section grouping (max 4 groups) ===")

    if not drafted_grouping and not all_tag_names:
        print("  [Notice] No tags to group for this resume.")
        return {}

    current_grouping = dict(drafted_grouping)

    while True:
        for label, tags in current_grouping.items():
            print(f"  [{label}]: {', '.join(tags)}")

        raw = input("\n  Accept this grouping? (Enter to accept and continue / 'r' or 'r <text>' to regenerate): ").strip()
        decision = parse_review_input(raw)

        if decision.action == "regenerate":
            if tags_with_category is None or job_description is None:
                print(
                    "  [Warning] Cannot regenerate -- this grouping was confirmed without "
                    "tags_with_category/job_description context. Continuing with the current draft."
                )
                break

            print(
                "  [Notice] Regenerating replaces the current draft entirely -- any manual "
                "group edits made so far in this session will be discarded."
            )
            current_grouping = draft_skill_grouping(tags_with_category, job_description, steering_note=decision.value)
            continue

        if decision.action == "accept":
            break

        print(f"  [Error] {decision.value}")

    grouping = _resolve_unplaced_tags(current_grouping, all_tag_names)
    grouping = _prompt_add_new_groups(grouping)

    print("\n=== Final skills-section grouping ===")
    for label, tags in grouping.items():
        print(f"  [{label}]: {', '.join(tags)}")

    return grouping


def group_skills_for_jd(tags_with_category, job_description):
    all_tag_names = {t for t, _ in tags_with_category}
    drafted = draft_skill_grouping(tags_with_category, job_description)
    return confirm_skill_grouping(
        drafted, all_tag_names,
        tags_with_category=tags_with_category, job_description=job_description,
    )
