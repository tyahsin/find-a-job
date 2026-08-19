"""
skill_grouper.py
Version: v1 (2026-08-16)

Solves the skill-section grouping mismatch problem: TagORM.tag_category
(skill/domain/job_function) is a DATA-LAYER classification, chosen for
governance/relevance-matching purposes (experience_matcher.py's tag
signal). It is NOT designed to match how a specific job posting's
audience expects skills to be grouped on a resume (e.g. "Programming
Languages / Cloud & Infra / Data Tools" for a Data Engineer JD, vs.
"Frontend / Backend / DevOps" for a different JD).

This module adds a PRESENTATION-LAYER regrouping step: a single batch LLM
call takes the flat set of tag_names already selected for this resume
(via the same claim_ids used elsewhere in the pipeline), plus their
original tag_category as a soft hint (not a hard constraint), plus the
job description, and asks the LLM to propose a JD-appropriate grouping
for this resume -- capped at 4 groups.

Design decisions (per user confirmation, 2026-08-16):
1. MANDATORY human confirmation gate -- same safety pattern as
experience_rewriter.py's _confirm_one() / summary_writer.py's
confirm_summary().
2. Hard cap: at most 4 groups. Prevents over-fragmentation.
3. tag_category IS passed to the LLM as a soft signal (not enforced) --
prevents unstable groupings and skill/domain conflation across runs
(rejecting a zero-context design was an explicit user decision: without
this hint, the same tag set could be split differently across runs of
the same JD, and terms like "Machine Learning" are prone to being
miscast as either a tool or a domain depending on phrasing alone).
4. NO automatic fallback "Other" group for tags the LLM failed to place.
Instead, any unplaced tag triggers an INTERACTIVE per-tag CLI prompt at
confirmation time, where the user assigns it to an existing group or a
newly-typed group name. Silently bucketing into "Other" was explicitly
rejected by the user -- every unplaced tag requires an explicit human
decision, not a catch-all default.
5. A separate interactive CLI step lets the user add brand-new groups
beyond what the LLM proposed (e.g. splitting one drafted group into two,
or introducing a group the LLM didn't think of), independent of whether
there are any unplaced tags.

Safety rules (same family as experience_rewriter.py / education_selector.py):
- The LLM may ONLY re-bucket existing tag_name strings into new group
labels. It may NOT invent, rename, merge the text of, or drop any
tag_name. Every tag_name given in the input must appear in exactly one
final group after the human confirmation step (LLM output need not be
complete -- confirm_skill_grouping() closes any gaps interactively).
- English-only prompts and output, per the project-wide LANGUAGE POLICY.

Consumed by resume_builder.py v13's build_skills_section(), which now
requires a job_description parameter to pass through to this module,
replacing the previous hardcoded label_map = {"skill": "Skills", "domain":
"Domain Knowledge", "job_function": "Functions"} static grouping.
"""

from __future__ import annotations

import json
import re

from core.llm_client import get_llm_client

MAX_GROUPS = 4

GROUPING_SYSTEM_PROMPT = f"""You are a resume skills-section organizer. You will be given a target job
description and a flat list of skill/knowledge tags the candidate has
already confirmed apply to this resume. Each tag also comes with its
original data-layer category (skill / domain / job_function) as a hint,
but that category system is NOT meant for resume presentation -- it is
an internal classification and may not match how a reader of this
specific job posting would expect skills to be grouped.

Your task: propose new group labels and assign every tag to exactly one
group, choosing groupings that make sense for THIS job description's
domain and terminology (e.g. "Programming Languages", "Cloud & Infrastructure",
"Data Tools", "Leadership & Collaboration" -- choose labels appropriate
to the actual tags and JD, do not just copy this example).

Strict rules (must follow):
1. HARD LIMIT: produce AT MOST {MAX_GROUPS} groups. Fewer is fine if that
better fits the tags given -- do not pad to reach {MAX_GROUPS}.
2. Try to place every tag in the input list into exactly one output
group. If a tag genuinely does not fit any sensible group, it is
better to leave it out than to force it somewhere wrong -- a human
will resolve any leftover tags afterward.
3. Do NOT change the text of any tag -- copy each tag_name exactly as
given, verbatim.
4. The original category (skill/domain/job_function) is a hint about
what KIND of thing each tag is -- use it to avoid conflating skills
with domain knowledge, but you are free to split, merge, or relabel
across that boundary if the job description calls for it.
5. Choose group labels that are natural, resume-appropriate section
headers (Title Case, 1-4 words each) -- not generic labels like
"Category 1".
6. Output must be in English, regardless of input language.

Output format: output ONLY a valid JSON object mapping each group label
(string) to a list of tag_name strings, and nothing else -- no preamble,
no explanation, no markdown code fences.
Example shape (illustrative only, do not copy these exact labels/tags):
{{"Programming Languages": ["Python", "SQL"], "Cloud & Infrastructure": ["Docker", "AWS"]}}"""

GROUPING_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Tags to group ({count} total)
{tags_block}

Output the grouped JSON (at most {max_groups} groups):"""


def _format_tags_block(tags_with_category: list[tuple[str, str]]) -> str:
    lines = []
    for tag_name, tag_category in tags_with_category:
        lines.append(f"- {tag_name} (original category: {tag_category})")
    return "\n".join(lines)


def _parse_grouping_response(raw_response: str, all_tag_names: set[str]) -> dict[str, list[str]]:
    """Tolerant JSON parsing + validation: unknown tags the LLM might have
    invented are dropped. Tags the LLM left out are intentionally NOT
    auto-bucketed here -- they are surfaced to confirm_skill_grouping()
    as unplaced, for interactive resolution (no silent 'Other' bucket)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    seen: set[str] = set()
    result: dict[str, list[str]] = {}
    for label, tags in list(parsed.items())[:MAX_GROUPS]:
        if not isinstance(tags, list):
            continue
        valid_tags = [t for t in tags if isinstance(t, str) and t in all_tag_names and t not in seen]
        if valid_tags:
            result[str(label)] = valid_tags
            seen.update(valid_tags)

    return result


def draft_skill_grouping(
    tags_with_category: list[tuple[str, str]], job_description: str
) -> dict[str, list[str]]:
    """Single batch LLM call. Returns {group_label: [tag_name, ...]},
    capped at MAX_GROUPS groups. May not cover every input tag -- any
    gaps are resolved interactively in confirm_skill_grouping()."""
    if not tags_with_category:
        return {}

    all_tag_names = {t for t, _ in tags_with_category}

    user_prompt = GROUPING_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(tags_with_category),
        tags_block=_format_tags_block(tags_with_category),
        max_groups=MAX_GROUPS,
    )

    response = get_llm_client().generate_text(
        system_prompt=GROUPING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=800,
    )

    return _parse_grouping_response(response.text, all_tag_names)


def _prompt_assign_unplaced_tag(tag_name: str, group_labels: list[str]) -> str:
    """Interactive per-tag resolution for a single unplaced tag. Lets the
    user pick an existing group by number, or type a brand-new group
    name on the spot."""
    print(f"\n Unplaced tag: '{tag_name}'")
    for i, label in enumerate(group_labels, 1):
        print(f"   [{i}] {label}")
    print(f"   [{len(group_labels) + 1}] Create a new group for this tag")

    choice = input(" Assign to which group? (enter a number, or type a new group name): ").strip()

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(group_labels):
            return group_labels[idx - 1]
        if idx == len(group_labels) + 1:
            new_label = input(" New group name: ").strip()
            return new_label or "Skills"

    return choice if choice else "Skills"


def _resolve_unplaced_tags(
    grouping: dict[str, list[str]], all_tag_names: set[str]
) -> dict[str, list[str]]:
    """Finds any tag from the original input that the LLM didn't place
    into a group, and resolves each one interactively -- no automatic
    'Other' fallback."""
    placed = {t for tags in grouping.values() for t in tags}
    unplaced = sorted(all_tag_names - placed)

    if not unplaced:
        return grouping

    print(f"\n=== {len(unplaced)} tag(s) were not placed into a group by the draft ===")
    for tag_name in unplaced:
        target_label = _prompt_assign_unplaced_tag(tag_name, list(grouping.keys()))
        grouping.setdefault(target_label, []).append(tag_name)

    return grouping


def _prompt_add_new_groups(grouping: dict[str, list[str]]) -> dict[str, list[str]]:
    """Lets the user add brand-new groups on top of the (already-resolved)
    grouping, e.g. splitting an existing group into two. Independent of
    whether there were any unplaced tags."""
    choice = input(
        "\n Add a new group by moving tags out of existing ones? (y/N): "
    ).strip().lower()

    if choice != "y":
        return grouping

    while True:
        new_label = input(" New group name (leave blank to stop adding groups): ").strip()
        if not new_label:
            break

        print(" Current tags by group:")
        for label, tags in grouping.items():
            print(f"   [{label}]: {', '.join(tags)}")

        raw_tags = input(
            f" Which tags should move into '{new_label}'? (comma-separated, exact names): "
        ).strip()
        tags_to_move = [t.strip() for t in raw_tags.split(",") if t.strip()]

        moved: list[str] = []
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

        print(f" -> Created '{new_label}' with: {', '.join(moved) if moved else '(no tags matched)'}")

    return grouping


def confirm_skill_grouping(
    drafted_grouping: dict[str, list[str]], all_tag_names: set[str]
) -> dict[str, list[str]]:
    """Mandatory human confirmation gate. Three steps, in order:
    1. Display the drafted grouping.
    2. Interactively resolve any tag the LLM left unplaced (no auto
    'Other' bucket -- user explicitly assigns each one).
    3. Optionally let the user carve out additional new groups.
    """
    print("\n=== Proposed skills-section grouping (max 4 groups) ===")

    if not drafted_grouping and not all_tag_names:
        print(" [Notice] No tags to group for this resume.")
        return {}

    for label, tags in drafted_grouping.items():
        print(f" [{label}]: {', '.join(tags)}")

    grouping = _resolve_unplaced_tags(dict(drafted_grouping), all_tag_names)
    grouping = _prompt_add_new_groups(grouping)

    print("\n=== Final skills-section grouping ===")
    for label, tags in grouping.items():
        print(f" [{label}]: {', '.join(tags)}")

    return grouping


def group_skills_for_jd(
    tags_with_category: list[tuple[str, str]], job_description: str
) -> dict[str, list[str]]:
    """Main entry point: single LLM draft + mandatory interactive human
    confirmation (including resolving unplaced tags and optionally adding
    new groups). Returns the final {group_label: [tag_name, ...]} mapping
    to render into the resume's skills section."""
    all_tag_names = {t for t, _ in tags_with_category}
    drafted = draft_skill_grouping(tags_with_category, job_description)
    return confirm_skill_grouping(drafted, all_tag_names)
