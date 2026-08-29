"""
skill_grouper.py
Version: v4 (2026-08-29)

Changelog:
- v4 (2026-08-29): Reworked the whole flow.
  (1) Programming languages are extracted FIRST via a dedicated,
      JD-independent LLM call, THEN unioned with a hardcoded language-name
      fallback (KNOWN_PROGRAMMING_LANGUAGES) so a language the LLM misses is
      still forced into the guaranteed "Programming" group (which is NOT
      counted toward MAX_GROUPS).
  (2) The remaining (non-programming) tags are grouped by the existing LLM
      grouping call; programming tags are removed from that call's input.
  (3) Every group is rank-ordered by deterministic JD relevance
      (gencv/tag_ranker.py) and capped to top TOP_K_PER_GROUP (10) -- applied
      PER GROUP after grouping (gentle cap, preserves breadth).
  (4) Confirmation is TWO layers; the ONLY actions are accept (Enter) and
      regenerate ('r' / 'r <text>'). The old unplaced-tag resolution and
      add-new-group steps are removed.
      - Layer 1 confirms the GROUP SKELETON; 'r' re-rolls the non-Programming
        grouping wholesale, 'r <text>' steers it.
      - Layer 2 confirms EACH group in order; 'r' re-selects that group's
        tags from a pool of (its own current tags + UNPLACED tags only --
        never tags already assigned to another group), 'r <text>' steers it
        and may inject tags not in the DB (cv.yaml only, never written back).

Safety: the LLM may ONLY re-bucket / re-select existing tag_name strings.
The one exception is Layer-2 'r <text>', where the user explicitly injects a
tag not in the DB -- that tag reaches cv.yaml only and is never persisted.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from __future__ import annotations

import json
import re

from core.llm_client import get_llm_client
from core.llm_reviewer import parse_review_input
from gencv.tag_ranker import rank_tags_by_jd

MAX_GROUPS = 4            # applies to NON-programming groups only
TOP_K_PER_GROUP = 10
PROGRAMMING_LABEL = "Programming"

# Deterministic fallback for the guaranteed Programming group. Tuned to the
# target roles: data science / data analysis / AI engineering / forward
# deployed engineer in finance & tech. Matched case-insensitively against the
# FULL tag_name (exact, not substring -- "c" matches only a tag literally
# named "C", never "C++"). Extend freely as your tag vocabulary grows.
KNOWN_PROGRAMMING_LANGUAGES = {
    "python", "r", "sql", "scala", "java", "c", "c++", "c#",
    "go", "golang", "rust", "julia", "matlab", "sas", "stata",
    "bash", "shell", "powershell", "javascript", "typescript",
    "kotlin", "vba", "perl",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROGRAMMING_EXTRACTION_SYSTEM_PROMPT = """You are given a flat list of skill tags. Identify which ones are
PROGRAMMING or SCRIPTING LANGUAGES (e.g. Python, Java, C++, C#, Go, R, SQL,
Bash/shell). Query languages such as SQL count. Markup/style languages
(HTML, CSS) do NOT count. Frameworks, libraries, tools, platforms, and
methodologies do NOT count.

Output format: output ONLY a JSON array of the exact tag_name strings that
are programming/scripting languages, copied verbatim, and nothing else --
no preamble, no markdown code fences. If none qualify, output []."""

PROGRAMMING_EXTRACTION_USER_TEMPLATE = """# Tags
{tags_block}

Output the JSON array of programming/scripting language tag_names:"""


GROUPING_SYSTEM_PROMPT = """You are a resume skills-section organizer. You will be given a target job
description and a flat list of skill/knowledge tags the candidate has
already confirmed apply to this resume. Each tag also comes with its
original data-layer category (tool / capability / context) as a hint, but
that category system is NOT meant for resume presentation.

Your task: propose group labels and assign every tag to exactly one group.

GROUPING AXIS (most important): group by TECHNOLOGY TYPE and TOOL/PACKAGE
ECOSYSTEM, the way a technical resume's skills section reads -- NOT by soft
skill, seniority, or generic competency. Group labels must name a *kind of
technology or technical area*, so a reader instantly sees the tech stack.

Good group labels (tool/package/tech-area oriented, illustrative):
  "ML Frameworks"        -> PyTorch, JAX, TensorFlow, ONNX
  "Infrastructure"       -> Kubernetes, Ray, AWS, GCP, distributed training
  "Data & Databases"     -> PostgreSQL, Spark, dbt, Snowflake
  "Visualization & BI"   -> Tableau, Power BI, Plotly
  "Research Areas"       -> model compression, efficient inference, multi-agent RL
(the last one shows where non-tool technical METHODS/AREAS go -- a
technology-oriented label, still not a soft-skill label)

Bad group labels (do NOT produce these): "Data-Driven Decision Making",
"Cross-functional Collaboration", "Problem Solving", "Leadership",
"Core Competencies", "Soft Skills" -- these are competencies, not tech areas.

{steering_note_block}

Strict rules (must follow):
1. HARD LIMIT: produce AT MOST {max_groups} groups. Fewer is fine -- do not
   pad to reach {max_groups}.
2. Do NOT create a "Programming" or "Languages" group -- programming
   languages are handled separately and are NOT in the list below.
3. Every group label must be technology/tool/package/technical-area
   oriented (see GROUPING AXIS above), never a soft skill or generic
   competency.
4. capability-category tags (transferable technical METHODS, e.g. "Feature
   Engineering", "Statistical Analysis") are technical and DO belong here --
   place them under a technical-AREA label such as "Methods", "Techniques",
   or "Research Areas", NOT under a soft-skill label.
5. Try to place every tag into exactly one group. If a tag genuinely does
   not fit, leaving it out is better than forcing it.
6. Do NOT change the text of any tag -- copy each tag_name verbatim.
7. Choose natural, resume-appropriate section headers (Title Case, 1-4 words).
8. Output must be in English.

Output format: output ONLY a valid JSON object mapping each group label to a
list of tag_name strings, and nothing else -- no preamble, no code fences.
Example shape (illustrative): {{"ML Frameworks": ["PyTorch", "JAX"], "Infrastructure": ["Kubernetes", "AWS"]}}"""

GROUPING_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Tags to group ({count} total)
{tags_block}

Output the grouped JSON (at most {max_groups} groups):"""


REGROUP_ONE_SYSTEM_PROMPT = """You are refining ONE group of a resume skills section. You will be given a
target job description, the group's label, and a POOL of available tags.
Select the tags from the pool that best belong under this group's label for
THIS job description.

{steering_note_block}

Strict rules:
1. Copy each chosen tag_name exactly, verbatim, from the pool.
2. Do NOT invent tags. (Exception: if the user's steering note explicitly
   asks to add a specific tag not in the pool, include it.)
3. Choose only tags that genuinely fit this group's label.
4. Output must be in English.

Output format: output ONLY a JSON array of the chosen tag_name strings, and
nothing else -- no preamble, no code fences. If none fit, output []."""

REGROUP_ONE_USER_TEMPLATE = """# Job Description
{job_description}

# Group label
{label}

# Available tag pool ({count} total)
{pool_block}

Output the JSON array of tag_names for the "{label}" group:"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _strip_fence(raw: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


def _parse_tag_list(raw_response: str) -> list[str]:
    try:
        parsed = json.loads(_strip_fence(raw_response))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return []


def _parse_grouping_response(raw_response: str, all_tag_names: set) -> dict:
    try:
        parsed = json.loads(_strip_fence(raw_response))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    seen = set()
    result = {}
    for label, tags in list(parsed.items())[:MAX_GROUPS]:
        if not isinstance(tags, list):
            continue
        valid = [t for t in tags if isinstance(t, str) and t in all_tag_names and t not in seen]
        if valid:
            result[str(label)] = valid
            seen.update(valid)
    return result


# ---------------------------------------------------------------------------
# LLM steps
# ---------------------------------------------------------------------------

def extract_programming_tags(tags_with_category) -> set:
    """LLM call A -- JD-independent. Returns the subset of tag_names the LLM
    judges to be programming/scripting languages (only exact DB tags kept).
    The hardcoded fallback union happens in group_skills_for_jd()."""
    names = [t for t, _ in tags_with_category]
    if not names:
        return set()
    response = get_llm_client().generate_text(
        system_prompt=PROGRAMMING_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=PROGRAMMING_EXTRACTION_USER_TEMPLATE.format(
            tags_block="\n".join(f"- {t}" for t in names)
        ),
        temperature=0.0,
        max_tokens=300,
    )
    name_set = set(names)
    return {t for t in _parse_tag_list(response.text) if t in name_set}


def draft_skill_grouping(remaining_with_category, job_description, steering_note=None) -> dict:
    """Groups the NON-programming tags via one LLM call."""
    if not remaining_with_category:
        return {}
    all_names = {t for t, _ in remaining_with_category}
    steering_block = (
        "The user was not satisfied with a previous grouping and gave this "
        "steering note -- treat it as an additional preference on top of the "
        "rules above (never as license to rename, invent, or drop a tag):\n"
        f'"{steering_note}"'
    ) if steering_note else ""
    tags_block = "\n".join(f"- {t} (original category: {c})" for t, c in remaining_with_category)
    response = get_llm_client().generate_text(
        system_prompt=GROUPING_SYSTEM_PROMPT.format(
            steering_note_block=steering_block, max_groups=MAX_GROUPS
        ),
        user_prompt=GROUPING_USER_PROMPT_TEMPLATE.format(
            job_description=job_description, count=len(remaining_with_category),
            tags_block=tags_block, max_groups=MAX_GROUPS,
        ),
        temperature=0.2 if not steering_note else 0.4,
        max_tokens=800,
    )
    return _parse_grouping_response(response.text, all_names)


def _regenerate_one_group(label, pool_with_category, job_description,
                          steering_note, rank_index) -> list:
    """Layer-2 'r' -- re-select ONE group's tags from the given pool via an
    independent LLM call. 'r <text>' (steering_note present) may inject tags
    not in the pool/DB."""
    pool_names = {t for t, _ in pool_with_category}
    steering_block = (f'User steering note: "{steering_note}"' if steering_note else "")
    response = get_llm_client().generate_text(
        system_prompt=REGROUP_ONE_SYSTEM_PROMPT.format(steering_note_block=steering_block),
        user_prompt=REGROUP_ONE_USER_TEMPLATE.format(
            job_description=job_description, label=label,
            count=len(pool_with_category),
            pool_block="\n".join(f"- {t} (category: {c})" for t, c in pool_with_category),
        ),
        temperature=0.3,
        max_tokens=400,
    )
    picked = _parse_tag_list(response.text)
    result = []
    for t in picked:
        if t in pool_names:
            result.append(t)
        elif steering_note:            # explicit user steer -> allow non-DB tag
            result.append(t)
    return _rank_and_cap(result, rank_index, TOP_K_PER_GROUP)


# ---------------------------------------------------------------------------
# Ranking / capping / layout
# ---------------------------------------------------------------------------

def _rank_and_cap(tags, rank_index, k) -> list:
    """Sort by global JD-relevance rank (unknown/injected tags sink to the
    end in given order), take top k."""
    known = [t for t in tags if t in rank_index]
    unknown = [t for t in tags if t not in rank_index]
    known.sort(key=lambda t: rank_index[t])
    return (known + unknown)[:k]


def _build_layout(programming, non_prog_grouping, rank_index) -> dict:
    layout = {}
    if programming:
        layout[PROGRAMMING_LABEL] = _rank_and_cap(list(programming), rank_index, TOP_K_PER_GROUP)
    for label, tags in non_prog_grouping.items():
        layout[label] = _rank_and_cap(tags, rank_index, TOP_K_PER_GROUP)
    return layout


# ---------------------------------------------------------------------------
# Confirmation (two layers; accept + regenerate only)
# ---------------------------------------------------------------------------

def confirm_skill_layout(layout, programming, tags_with_category, job_description, rank_index) -> dict:
    category_of = {t: c for t, c in tags_with_category}
    all_names = [t for t, _ in tags_with_category]
    programming = set(programming)

    # ---- Layer 1: confirm the group skeleton ----
    while True:
        print("\n=== Proposed skills-section layout ===")
        for label, tags in layout.items():
            print(f"  [{label}]: {', '.join(tags)}")
        raw = input(
            "\n  Accept these groups? (Enter to accept / 'r' or 'r <text>' "
            "to regenerate the whole layout): "
        ).strip()
        decision = parse_review_input(raw)
        if decision.action == "accept":
            break
        if decision.action == "regenerate":
            remaining = [(t, c) for t, c in tags_with_category if t not in programming]
            drafted = draft_skill_grouping(remaining, job_description, steering_note=decision.value)
            layout = _build_layout(programming, drafted, rank_index)
            continue
        print(f"  [Error] {decision.value}")

    # ---- Layer 2: confirm each group in order ----
    labels = list(layout.keys())
    locked = set()
    finalized = {}
    for label in labels:
        current = [t for t in layout[label] if t not in locked]
        while True:
            print(f"\n  [{label}]: {', '.join(current) if current else '(empty)'}")
            raw = input(
                "  Accept this group? (Enter to accept / 'r' or 'r <text>' "
                "to regenerate this group): "
            ).strip()
            decision = parse_review_input(raw)
            if decision.action == "accept":
                finalized[label] = current
                locked.update(current)
                layout[label] = current           # keep layout in sync
                break
            if decision.action == "regenerate":
                # Option B pool: this group's own tags + UNPLACED tags only.
                # Never steal tags assigned to another group (confirmed or not).
                placed_elsewhere = set()
                for other in labels:
                    if other != label:
                        placed_elsewhere.update(layout[other])
                disallow = placed_elsewhere | set(current)
                if label != PROGRAMMING_LABEL:
                    disallow |= programming        # languages never leak into non-prog groups
                unplaced = [(t, category_of[t]) for t in all_names if t not in disallow]
                pool = [(t, category_of.get(t, "injected")) for t in current] + unplaced
                current = _regenerate_one_group(label, pool, job_description, decision.value, rank_index)
                current = [t for t in current if t not in locked]
                continue
            print(f"  [Error] {decision.value}")

    finalized = {label: tags for label, tags in finalized.items() if tags}
    print("\n=== Final skills-section layout ===")
    for label, tags in finalized.items():
        print(f"  [{label}]: {', '.join(tags)}")
    return finalized


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def group_skills_for_jd(tags_with_category, job_description) -> dict:
    if not tags_with_category:
        return {}
    all_names = [t for t, _ in tags_with_category]

    # 1. Programming: LLM extraction UNION hardcoded-name fallback (per-tag
    #    top-up -- a known language the LLM missed is still forced in).
    programming = extract_programming_tags(tags_with_category)
    programming |= {t for t in all_names if t.lower() in KNOWN_PROGRAMMING_LANGUAGES}

    # 2. Group the remaining (non-programming) tags.
    remaining = [(t, c) for t, c in tags_with_category if t not in programming]
    drafted = draft_skill_grouping(remaining, job_description)

    # 3. Deterministic global relevance order (within-group sort + top-10).
    ranked = rank_tags_by_jd(all_names, job_description)
    rank_index = {name: i for i, (name, _) in enumerate(ranked)}

    # 4. Assemble layout (Programming first, guaranteed, excluded from MAX_GROUPS).
    layout = _build_layout(programming, drafted, rank_index)

    # 5. Two-layer confirmation.
    return confirm_skill_layout(layout, programming, tags_with_category, job_description, rank_index)