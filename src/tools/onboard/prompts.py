"""
onboard/prompts.py
Version: v1.2 (2026-08-10)

Changelog:
- v1.2 (2026-08-10): Full English-only pass. EVIDENCE_TYPE_TO_LEVEL and
  METRIC_CONFIDENCE_OPTIONS (previously left in Chinese as "human-selected,
  never LLM output") converted to English per the project-wide decision
  that the ENTIRE system -- every prompt and every interface element --
  must be English-only, with no exceptions carved out for fields that
  happen not to be LLM output.
- v1.1 (2026-08-10, earlier same-day revision): _BASE_SYSTEM_RULES,
  TAG_SYSTEM_PROMPT, and METRIC_EXTRACTION_SYSTEM_PROMPT (+
  VALUE_TYPE_OPTIONS/METRIC_TYPE_OPTIONS) converted to English. Fixed the
  root contradiction where _BASE_SYSTEM_RULES demanded Chinese output while
  individual prompts separately bolted on "must be in English".
- v1.0 (2026-08-04, inferred): Original Chinese-language prompt templates.

Prompt templates used by the onboard CLI flows to draft:
- position.responsibility (from a source file OR from a spoken/typed summary)
- project.project_context (from a short user-provided description)
- action.description (what was done)
- claim.description (what result followed from the action -- deliberately
  scoped AWAY from action's territory, see RESULT_CLAIM_SYSTEM_PROMPT below)

Design principle (matches the project's governance rules): the LLM only
rephrases/organizes what the user already provided. It must NOT invent
responsibilities, scope, dates, or achievements that aren't present in the
input. All drafts are shown to the user for confirmation/edit before being
persisted (precompute-once, human-reviewed).

LANGUAGE POLICY (final, v1.2): the entire onboard/CV-generation system --
every LLM prompt AND every user-facing interface element (CLI prompts,
menu options, fixed-choice vocabularies) -- is English-only. This covers:
  - the five narrative description prompts (_BASE_SYSTEM_RULES)
  - tag_name in TAG_SYSTEM_PROMPT
  - all three fields extracted by METRIC_EXTRACTION_SYSTEM_PROMPT
  - VALUE_TYPE_OPTIONS / METRIC_TYPE_OPTIONS (LLM output AND human-selected
    CLI menu, same list)
  - EVIDENCE_TYPE_TO_LEVEL (human-selected CLI menu)
  - METRIC_CONFIDENCE_OPTIONS (human-selected CLI menu)
  - ROLE_SCOPE_OPTIONS (already English, unchanged)

MIGRATION NOTE: EVIDENCE_TYPE_TO_LEVEL, VALUE_TYPE_OPTIONS/
METRIC_TYPE_OPTIONS, and METRIC_CONFIDENCE_OPTIONS all changed their
string values from Chinese to English across v1.1/v1.2. Any already-
persisted rows (evidence.evidence_type, claim_metric.metric_type,
claim_metric.metric_confidence) using the old Chinese values will not
match these new English option lists or parse_metric_response's
valid_types set -- check/backfill existing data if any exists before
relying on exact-match logic against it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared system-level guardrails
# ---------------------------------------------------------------------------

_BASE_SYSTEM_RULES = """You are a writing assistant that helps organize a person's career and
education experience for resume/CV purposes. Your task is to take the raw
material the user provides and turn it into a concise, clear, professional
ENGLISH summary.

Strict rules (must follow):
1. Do not add any concrete fact, number, scope of responsibility, or
   achievement that does not already appear in the user-provided material.
2. Do not rewrite uncertain or vague statements into more confident or
   exaggerated ones.
3. Do not infer or extend any skill, responsibility, or scope of
   contribution the user did not explicitly mention.
4. You may only: reorganize sentences, polish word choice, remove filler
   words, unify tone, and improve readability.
5. If the original material is too short or vague, the summary should stay
   equally short/vague -- do not "fill in the gaps" yourself.
6. Output must be written in English, no matter what language the user's
   input material is written in.

Output format: output ONLY the summary text itself, plain text, with no
preamble, heading, or explanatory text.
"""

# ---------------------------------------------------------------------------
# position.responsibility
# ---------------------------------------------------------------------------

RESPONSIBILITY_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
Purpose of this summary: describe the scope of responsibilities the user
held during a given position (position.responsibility). It should be an
objective description of duties, not an achievement narrative (achievements
are handled separately elsewhere).
Suggested length: 2-4 sentences.
"""


def build_responsibility_prompt_from_source(source_text: str) -> str:
    """
    Use when the user provided a source file (JD, offer letter, etc.)
    via responsibility_source. The LLM summarizes the ACTUAL document
    content -- it must not add responsibilities beyond what's written.
    """
    return f"""Below is the raw document content provided by the user (may be a
position description, offer letter, etc.). Please summarize the part
about "scope of responsibilities" into a concise summary:

---
{source_text}
---

Output the summary (plain text, no preamble):"""


def build_responsibility_prompt_from_description(user_description: str) -> str:
    """
    Use when the user has no source file and instead typed/spoke a rough
    description of their responsibilities. The LLM only polishes wording;
    it must not add scope beyond what the user described.
    """
    return f"""The user described the scope of responsibilities for this position in
their own words; the content may be casual or brief. Please organize it
into a concise, professional summary without adding anything the user
did not mention:

---
{user_description}
---

Output the summary (plain text, no preamble):"""

# ---------------------------------------------------------------------------
# degree.degree_description
# ---------------------------------------------------------------------------

DEGREE_DESCRIPTION_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
Purpose of this summary: describe the scope/focus of a degree program
(degree.degree_description). This should be an objective description of
what the program covered or focused on, not an achievement narrative.
Stress more on the curriculum where  
Suggested length: 2 sentences at most and less than 25 words.
"""


def build_degree_description_prompt_from_source(source_text: str) -> str:
    """Use when the user provided a source file (program syllabus,
    transcript, degree description page, etc.) via
    degree_description_source. The LLM summarizes the ACTUAL document
    content -- it must not add scope beyond what's written."""
    return f"""Below is the raw document content provided by the user (may be a
program syllabus, transcript excerpt, or degree description page). Please
summarize the part about the program's focus/scope into a concise summary:

---
{source_text}
---

Output the summary (plain text, no preamble):"""


def build_degree_description_prompt_from_description(user_description: str) -> str:
    """Use when the user has no source file and instead typed a rough
    description of the program's focus. The LLM only polishes wording; it
    must not add scope beyond what the user described."""
    return f"""The user described the focus/scope of this degree program in their own
words; the content may be casual or brief. Please organize it into a
concise, professional summary without adding anything the user did not
mention:

---
{user_description}
---

Output the summary (plain text, no preamble):"""

# ---------------------------------------------------------------------------
# project.project_context
# ---------------------------------------------------------------------------

PROJECT_CONTEXT_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
Purpose of this summary: describe the background/context in which a
project arose (project.project_context) -- what situation produced this
project and what kind of project it is. This is NOT an outcome narrative
(outcomes are handled separately via problem/action/result_claim).
Suggested length: 1-3 sentences.
"""

def build_project_context_prompt_from_source(source_text: str) -> str:
    """
    Use for drafting project.project_context from a source file.
    The LLM summarizes the ACTUAL document
    content -- it must not add scope beyond what's written.
    """
    return f"""Below is the raw document content provided by the user. Please
organize it into a concise project background narrative:

---
{source_text}
---

Output the summary (plain text, no preamble):"""


def build_project_context_prompt_from_description(user_description: str) -> str:
    """
    Use for drafting project.project_context from the user's short,
    free-text description of the project's background.
    """
    return f"""The user briefly described the background/context of a project. Please
organize it into a concise project background narrative:

---
{user_description}
---

Output the summary (plain text, no preamble):"""


# ---------------------------------------------------------------------------
# problem.description
# ---------------------------------------------------------------------------

PROBLEM_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
Purpose of this summary: describe the concrete problem encountered within
this project (problem.description) -- what situation/difficulty/gap
existed at the time. Do not describe any solution (that's action's job).
Suggested length: 1-2 sentences.
"""


def build_problem_prompt(user_description: str) -> str:
    return f"""The user described a problem encountered in this project. Please organize
it into a concise problem statement:

---
{user_description}
---

Output the summary (plain text, no preamble):"""


# ---------------------------------------------------------------------------
# action.description
# ---------------------------------------------------------------------------

ACTION_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
Purpose of this summary: describe what the user actually DID about the
problem above (action.description) -- only the concrete action, method, or
intervention taken. Do not describe the outcome/impact of the action
(that's result_claim's job).
Suggested length: 1-2 sentences.
"""


def build_action_prompt(user_description: str) -> str:
    return f"""The user described the action taken to address this problem. Please
organize it into a concise action statement:

---
{user_description}
---

Output the summary (plain text, no preamble):"""


# ---------------------------------------------------------------------------
# claim.description
# ---------------------------------------------------------------------------
# NOTE: previously this prompt's wording overlapped semantically with
# action's wording, causing user confusion about which layer they were
# filling in. This version explicitly draws the boundary: description =
# what happened AFTER the action, not the action itself.

RESULT_CLAIM_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
Purpose of this summary: describe the RESULT that followed from a
specific action (claim.description). Describe only the result itself --
e.g. data changes, deliverables, impact, efficiency gains, being
accepted/adopted, etc. Do not repeat "what was done" (that's action's
job) -- only describe "what happened after it was done".
Suggested length: 1-2 sentences, as concise as possible.
"""


def build_result_claim_prompt(user_description: str) -> str:
    return f"""The user described the result produced by this action. Please organize
it into a concise result statement (describe only the result, do not
repeat the action itself):

---
{user_description}
---

Output the summary (plain text, no preamble):"""


# ---------------------------------------------------------------------------
# evidence_type -> evidence_level mapping (Problem 3)
# ---------------------------------------------------------------------------
# Rationale: evidence strength is a function of evidence TYPE, not a
# separate subjective judgment call. Letting the user pick L1-L4 freely
# leads to inconsistent self-grading over time (the same kind of evidence
# might be graded L2 today and L3 next month). Instead, the CLI presents a
# fixed menu of evidence types, and evidence_level is derived automatically.
#
# This is a human-selected fixed menu (never LLM output). Converted to
# English in v1.2 per the "whole system, no exceptions" policy.

# L1 = strongest (verifiable, third-party or system-of-record evidence)
# L4 = weakest (unverifiable, self-reported recollection)

EVIDENCE_TYPE_TO_LEVEL: dict[str, str] = {
    "Formal documentation (contract / official report / system screenshot)": "L1",
    "Third-party verification (manager review / client feedback / formal evaluation)": "L1",
    "Data record (analytics report / system log / dashboard)": "L2",
    "Internal team documentation (meeting notes / email / Slack conversation)": "L2",
    "Verbal account / recollection (no written record, memory only)": "L3",
    "Inference / estimate (reasonable inference without direct evidence)": "L4",
}

# ---------------------------------------------------------------------------
# tag generation (Problem 1 & 2 & 3 from Aug 2026 review)
# ---------------------------------------------------------------------------
# Design: tags are NOT typed by the user. Once problem/action/claim/evidence
# are all filled in for a given add-claim run, the LLM reads the FULL
# accumulated text of that run (already held in memory -- the user never
# re-pastes anything) and proposes a set of tags. The user then
# confirms/edits the proposed list (not a yes/no gate -- tagging itself is
# a mandatory step of add-claim, only the exact tag content is adjustable).
#
# LANGUAGE POLICY: tag_name must be English, no exceptions. Tags exist
# specifically to be matched against job descriptions (JD) at CV-generation
# time, and JDs are overwhelmingly in English for this user's target roles
# -- Chinese tags would silently fail to match and break the whole matching
# step.

TAG_SYSTEM_PROMPT = """You are an assistant that helps organize tags for a person's career
experience, for later matching against job descriptions (JD).


CRITICAL RULE -- read first: every tag_name you output MUST be written in
English. This applies no matter what language the problem/action/result
text below is written in. Do not output Chinese, pinyin, or any
non-English tag_name under any circumstance.


Other strict rules (must follow):
1. Tags must be capabilities, contexts, or tools that are actually
   demonstrated in the input content (including the background context) --
   do not invent anything that doesn't appear there. Background context
   may only be used to help infer "context" (e.g. company industry) or
   confirm a "tool" already implied by the text; it must NOT be used on
   its own to generate a "capability" tag (a capability tag must be backed
   by the problem/action/result text).
2. Each tag belongs to exactly one of three axes:
   - "capability": a transferable technical method or skill that would
     still apply if the specific tool were swapped for an equivalent
     alternative (e.g. "Feature Engineering", "Statistical Analysis",
     "Predictive Modeling", "Data Pipeline Development").
   - "context": the industry, application domain, or subject-matter area
     the work was applied to (e.g. "Fintech", "EdTech", "Sports
     Analytics", "Supply Chain").
   - "tool": a specific named programming language, framework, library,
     or platform mentioned or clearly implied in the text (e.g. "Python",
     "XGBoost", "Docker", "PostgreSQL", "AWS").
3. If a candidate tag could plausibly fit two axes, prefer classifying it
   as the MORE SPECIFIC axis: prefer "tool" over "capability" if a named
   product/library/language is involved; prefer "capability" over
   "context" if it describes a method rather than an industry.
4. Do not output a tag that is a generic umbrella term already implied by
   a more specific tag you are also outputting in the same batch (e.g. do
   not output both "Analytics" and "Sports Analytics" -- keep only the
   more specific one).
5. Tag names should be concise, generic, and reusable (e.g. "Feature
   Engineering" rather than "Used pandas to engineer features from raw
   data").
6. Produce at most 8 tags -- fewer and precise is better than many and
   inflated.


Output format: one tag per line, formatted as "tag_name | axis", where
tag_name is in English and axis is one of capability/context/tool.
No preamble, explanatory text, or extra punctuation.
"""


def build_tag_prompt(
    problem_description: str,
    action_description: str,
    description: str,
    evidence_type: str,
    static_context: str = "",
) -> str:
    """
    Builds the tag-generation prompt directly from the in-memory content
    already collected during this add-claim run -- problem/action/claim/
    evidence text is passed in as function arguments, not re-typed by the
    user.

    `static_context` carries the claim's linked static (relatively
    unchanging) data -- company/position or school/degree info -- resolved
    upstream from the claim -> action -> problem -> project -> position/
    degree -> company/school chain. Including this lets the LLM infer
    domain/job_function tags that are implied by *where* the work happened
    (e.g. "Fintech" from company industry, "Data Engineering" from job
    title) even when that context isn't explicitly repeated in the
    problem/action/result text. The caller (cli.py._build_static_context)
    formats this string in English too, so no Chinese leaks into this
    prompt.
    """
    context_block = (
        f"\\nBackground context (company/position or school/degree): {static_context}\\n"
        if static_context
        else ""
    )
    return f"""Below is the full content of this experience record:
{context_block}
Problem: {problem_description}
Action: {action_description}
Result: {description}
Evidence type: {evidence_type}

Based on the above (including background context), produce tags. Remember:
tag_name MUST be in English (format: tag_name | category):"""


def parse_tag_response(raw_text: str) -> list[tuple[str, str]]:
    """
    Parses lines like "Python | skill" into (tag_name, tag_category) tuples.
    Skips malformed lines rather than raising, since this is LLM output that
    the user will review/edit anyway before it's persisted.
    """
    valid_categories = {"skill", "domain", "job_function"}
    results: list[tuple[str, str]] = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name_part, _, category_part = line.partition("|")
        name = name_part.strip()
        category = category_part.strip().lower()
        if name and category in valid_categories:
            results.append((name, category))
    return results


# ---------------------------------------------------------------------------
# fixed-choice menus (Aug 2026 update: role_scope / value_type /
# metric_confidence should be selected from a list, not free-typed)
# ---------------------------------------------------------------------------
# ROLE_SCOPE_OPTIONS: human-selected fixed menu, already English.
#
# VALUE_TYPE_OPTIONS / METRIC_TYPE_OPTIONS: English (v1.1). This list is
# used BOTH as the set of values METRIC_EXTRACTION_SYSTEM_PROMPT asks the
# LLM to choose from, AND as the human-facing CLI menu in
# cli.py._review_metric_drafts() when the user manually adds a metric the
# LLM missed -- one list serves both call sites.
#
# METRIC_CONFIDENCE_OPTIONS: English (v1.2). Still never LLM output --
# confidence is always a manual human choice from this fixed CLI menu (LLM
# self-reported confidence is known to be poorly calibrated, see
# METRIC_EXTRACTION_SYSTEM_PROMPT rule 3).

ROLE_SCOPE_OPTIONS: list[str] = [
    "Led",
    "Collaborated",
    "Supported",
]

VALUE_TYPE_OPTIONS: list[str] = [
    "Efficiency Improvement",
    "Cost Saving",
    "Quality Improvement",
]

# metric_type reuses the same three categories plus "Other" for metrics
# that don't cleanly fit (this is exactly the gap that motivated splitting
# metrics out into their own claim_metric table).
METRIC_TYPE_OPTIONS: list[str] = VALUE_TYPE_OPTIONS + ["Other"]

METRIC_CONFIDENCE_OPTIONS: list[str] = [
    "Precise",
    "Estimated",
    "Rough impression",
]

# ---------------------------------------------------------------------------
# claim_metric extraction (Aug 2026 update)
# ---------------------------------------------------------------------------
# LLM only EXTRACTS metrics that are explicitly present in description
# text -- it must not estimate, invent, or self-report a confidence score
# (LLM self-reported confidence is known to be poorly calibrated). The user
# always reviews/edits the extracted draft, and metric_confidence is always
# a manual human selection from METRIC_CONFIDENCE_OPTIONS, never LLM output.
#
# LANGUAGE POLICY: all three extracted fields (type, label, value) are
# required to be English. type comes from the English METRIC_TYPE_OPTIONS
# above; label/value inherit English naturally once claim.description
# itself is generated in English, but the rule is stated explicitly here
# too as a defensive guarantee.

METRIC_EXTRACTION_SYSTEM_PROMPT = """Your task is to extract all quantifiable metrics from a result
description (description).

CRITICAL RULE -- read first: every field you output (type, label, value)
MUST be written in English, no matter what language the source description
is written in.

Other strict rules (must follow):
1. Only extract numbers/quantified statements explicitly present in the
   text -- do not guess, estimate, or invent data not written in the text.
2. Output three fields per metric: type, label, value.
   - type must be exactly one of: "Efficiency Improvement", "Cost Saving",
     "Quality Improvement", or "Other" -- use "Other" if it doesn't fit the
     first three. Do not translate or reword these four labels.
   - label: what the number actually represents (e.g. "processing time",
     "customer satisfaction"), described concisely, in English.
   - value: keep the original number/description intact but in English
     (e.g. "reduced from 5 days to 2 days", "about 15%").
3. Do not output a confidence level -- confidence is judged by the user
   manually and is not your concern.
4. If the text contains no quantifiable metric at all, return blank
   (output no lines).

Output format: one entry per line, separated by |: type|label|value
No preamble, explanatory text, or extra punctuation.
"""


def build_metric_extraction_prompt(description: str) -> str:
    return f"""Below is the result description (description) for this claim:

{description}

Extract all quantifiable metrics from it. Remember: type/label/value must
all be in English (format: type|label|value):"""


def parse_metric_response(raw_text: str) -> list[tuple[str, str, str]]:
    """
    Parses lines like "Efficiency Improvement|processing time|reduced from
    5 days to 2 days" into (metric_type, metric_label, metric_value)
    tuples. Skips malformed lines rather than raising, since this is an
    LLM draft the user reviews anyway.
    """
    valid_types = {"Efficiency Improvement", "Cost Saving", "Quality Improvement", "Other"}
    results: list[tuple[str, str, str]] = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or line.count("|") != 2:
            continue
        metric_type, metric_label, metric_value = (
            part.strip() for part in line.split("|")
        )
        if metric_type not in valid_types or not metric_label or not metric_value:
            continue
        results.append((metric_type, metric_label, metric_value))
    return results


CLAIMS_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
Purpose of this task: extract discrete, resume-ready achievement claims from 
a project or position description. Each claim represents ONE distinct accomplishment 
— not a duty description (that is handled separately in position.responsibility).


MECE extraction rules:
1. Mutually exclusive: each claim must describe a DIFFERENT accomplishment.
   If two potential claims share the same core action + same core result,
   merge them into one. Do not split a single outcome into multiple claims
   just because it touches multiple skills (e.g. "used Python and Docker to
   cut deploy time by 40%" is ONE claim, not two).
2. Collectively exhaustive: scan the ENTIRE input text before finalizing
   the list. Do not stop at the first few obvious claims — include claims
   with weaker evidence rather than omitting them, so the human reviewer
   decides what to keep.
3. Each claim must be verifiable against the source text. Do not infer or
   estimate information not present in the source text; use null instead.


Field separation rules (critical — do not let these three fields overlap
or restate each other):
- problem_description: describes the STATE BEFORE the action. It is the
  gap between expected and actual state that existed prior to any
  intervention. Contains no verbs describing what the person did.
- action_description: describes the INTERVENTION ITSELF. It is what the
  person concretely did, using an action verb. Contains no outcome or
  result language.
- claim_description: describes the STATE AFTER the action — the outcome
  or result. It is the resume-bullet-ready sentence combining context
  implicitly, but must not repeat the verb phrase already used in
  action_description, and must not restate the problem already captured
  in problem_description.
If you find yourself repeating the same clause across two of these three
fields, rewrite one of them to focus strictly on its assigned time frame
(before / during / after).


Output a list of JSON arrays. Each element must match this schema exactly:
{
    "problem_description": "the gap between expected and actual state before the action, or null if not stated in source",
    "stakeholder_impact": "who was affected and how significantly, or null if not stated in source",
    "baseline_metric": "measurable state before the action, or null if not stated in source",
    "action_description": "what was concretely done, action-verb start",
    "role_scope": "Led" | "Collaborated" | "Supported",
    "claim_description": "the result narrating how the problem is solved through the action, action-verb start"
}


Do not output any field not listed above. Do not output prose outside the
JSON array.


Lenght contraint: action_description and claim_description are 15-30 words
each. problem_description, stakeholder_impact, and baseline_metric are
concise (under 20 words) or null when not stated in source.

Output constraint: extract at most 8 claims per call. If the source text
could yield more than 8 distinct claims, prioritize those with the
strongest evidence and highest resume relevance; omit the weaker ones
rather than risk truncating output mid-object.
"""

def build_claims_prompt_from_source(source_text: str) -> dict:
    """
    Use for parsing project.project_context_source to list claims in MECE.
    The LLM summarizes the ACTUAL document
    content -- it must not add scope beyond what's written.
    """
    return f"""Below is the raw document content source provided by the user. Please
organize it into a list of Json arrays as system prompt suggested:

---
{source_text}
---

Output the list (plain text, no preamble):"""