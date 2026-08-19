"""
summary_writer.py
Version: v3 (2026-08-19)

Changelog:
- v3 (2026-08-19): Added a REGENERATE option to confirm_summary()'s
  confirmation gate, and migrated the gate to the shared
  core.llm_reviewer.parse_review_input() parser -- same single-line
  command contract as experience_rewriter.py v6 / claim_ranker.py v3 /
  skill_grouper.py v3: 'r' or 'r <text>' to regenerate, 'o <text>' for
  manual override, 'skip' to omit the summary entirely (unchanged
  behavior from v2, now expressed via the shared parser). Previously
  this module had no regenerate path at all -- the only options were
  accept, fully hand-type a replacement, or skip. Rationale: like
  skill_grouper.py's grouping draft, draft_summary() is already an LLM
  call at nonzero temperature, so a genuine re-roll (optionally guided
  by a steering note) can produce a materially different and useful
  result, without requiring the user to hand-write an entire replacement
  summary from scratch. The regeneration steering note is passed to the
  LLM as an additional preference layered on top of the existing
  SUMMARY_SYSTEM_PROMPT rules (see rule 9 below) -- it may only request a
  different presentation (tone, length, emphasis, which strengths to
  foreground) of the SAME confirmed bullets/education/tags, never a new
  fact, skill, or achievement beyond what was already given to
  draft_summary().
- v2 (2026-08-16): Added domain/job_function tags as an auxiliary signal
  input -- see module docstring for full rationale. domain/job_function
  tags are AUXILIARY BACKGROUND CONTEXT ONLY, never a forced-inclusion
  requirement.
- v1 (2026-08-15): Initial draft_summary()/draft_and_confirm_summary(),
  mandatory human confirmation gate, "bridge the gap only if real" logic.

Drafts a short professional summary (2-3 sentences, hard cap: 3) that sits
right after personal information and before all other sections in the
final resume -- a brief overview of the candidate + the job description,
bridging any gap between them ONLY IF ONE ACTUALLY EXISTS.

Design decisions:
- Single LLM call (not batch) per draft/regeneration -- there is only one
  summary to draft per generate_cv() run, and regeneration re-runs the
  same single call with an added steering note.
- MANDATORY human confirmation gate -- genuinely new generated text
  summarizing the entire resume.
- Input is deliberately restricted to CONFIRMED content only -- the final
  fused STAR bullets, the selected education entries, and the
  domain/job_function tags attached to the resume's included claims --
  never the full claim database.
- The "bridge the gap" instruction is explicitly conditional: if the
  candidate's confirmed experience already aligns well with the job
  description, the summary should NOT manufacture a gap-bridging
  narrative.
- Runs AFTER rewrite_and_confirm() and AFTER education selection, but
  BEFORE build_resume_yaml().

English-only prompt and output, per the project-wide LANGUAGE POLICY.
"""

from __future__ import annotations

from core.llm_client import get_llm_client
from core.llm_reviewer import parse_review_input

SUMMARY_SYSTEM_PROMPT = """You are a resume summary writer. You will be given a job description, a
set of already-confirmed resume bullet points (from the experience
section), a list of the candidate's education entries, and a list of the
candidate's domain/job-function tags (industry knowledge and role-type
labels the candidate has confirmed apply to them). Your task is to write
a short professional summary that will appear at the top of the resume,
right after the candidate's contact information.

Strict rules (must follow):
1. Base the summary ONLY on the bullet points, education entries, and
   domain/job-function tags given to you -- do not reference any
   experience, skill, or achievement that is not reflected in this
   material.
2. Do not add any concrete fact, number, or degree of achievement beyond
   what the bullet points already state.
3. The domain/job-function tags are AUXILIARY BACKGROUND CONTEXT ONLY --
   use them to inform word choice and framing, but you are NOT required
   to literally mention every tag, or any tag at all, in the summary
   text.
4. Judge whether there is an actual, meaningful gap between the
   candidate's background and the job description's requirements. If a
   real gap exists, briefly acknowledge the transition in a factual,
   non-defensive way. If NO meaningful gap exists, do NOT invent or imply
   one.
5. Write in active voice, third person is not used -- write as a direct,
   resume-style summary.
6. HARD LIMIT: the summary must be AT MOST 3 sentences. Prefer 2
   sentences when 2 are sufficient -- do not pad to reach 3.
7. No filler words, no generic claims not backed by the given material.
8. Output must be in English, regardless of the input language.
9. (v3) If you are given a REGENERATION STEERING NOTE, treat it as an
   additional preference for HOW to present the same confirmed material
   (tone, length, emphasis, which strengths to foreground) -- it may NOT
   be used to introduce a new fact, skill, or achievement beyond what the
   bullet points/education/tags already state, even if the note seems to
   ask for that. If the note conflicts with rules 1/2, follow rules 1/2
   instead.

Output format: output ONLY the summary text itself, plain text, no
preamble, no heading, no explanation, no quotation marks."""

SUMMARY_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Confirmed resume bullet points ({bullet_count} total)
{bullets_block}

# Education entries ({education_count} total)
{education_block}

# Domain / job-function tags ({tag_count} total, auxiliary context only)
{tags_block}

{steering_note_block}
Write the professional summary (at most 3 sentences):"""


def _format_bullets_block(final_bullets):
    if not final_bullets:
        return "(no experience bullets in this resume)"
    return "\n".join(f"- {b}" for b in final_bullets)


def _format_education_block(education_entries):
    if not education_entries:
        return "(no education entries in this resume)"
    lines = []
    for e in education_entries:
        parts = [e.get("institution", ""), e.get("degree", ""), e.get("area", "")]
        lines.append("- " + ", ".join(p for p in parts if p))
    return "\n".join(lines)


def _format_tags_block(domain_job_function_tags):
    if not domain_job_function_tags:
        return "(no domain/job-function tags available for this resume)"
    return ", ".join(sorted(set(domain_job_function_tags)))


def draft_summary(job_description, final_bullets, education_entries,
                   domain_job_function_tags=None, steering_note=None):
    """Single LLM call. Returns the drafted summary text (not yet
    confirmed -- see confirm_summary() for the mandatory human gate).

    (v3) steering_note is an optional regeneration instruction -- see
    module docstring and SUMMARY_SYSTEM_PROMPT rule 9 for why it is
    advisory-on-presentation only, never a forced-inclusion or
    new-fact requirement.
    """
    if steering_note:
        steering_note_block = (
            "# Regeneration steering note (presentation preference only -- see rule 9)\n"
            f'"{steering_note}"\n'
        )
    else:
        steering_note_block = ""

    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        bullet_count=len(final_bullets),
        bullets_block=_format_bullets_block(final_bullets),
        education_count=len(education_entries),
        education_block=_format_education_block(education_entries),
        tag_count=len(domain_job_function_tags) if domain_job_function_tags else 0,
        tags_block=_format_tags_block(domain_job_function_tags),
        steering_note_block=steering_note_block,
    )
    response = get_llm_client().generate_text(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3 if not steering_note else 0.5,
        max_tokens=200,
    )
    return response.text.strip()


def confirm_summary(drafted_summary, job_description=None, final_bullets=None,
                     education_entries=None, domain_job_function_tags=None):
    """(v3) Uses core.llm_reviewer.parse_review_input(). Commands:
    Enter=accept, 'r'/'r <text>'=regenerate, 'o <text>'=manual override,
    'skip'=omit the summary entirely. Plain 'o' or unrecognized input is
    invalid and re-prompts.

    job_description/final_bullets/education_entries/domain_job_function_tags
    are optional and only required if the user invokes regenerate (needed
    to re-call draft_summary()) -- if omitted, regenerate falls back to a
    warning and keeps the current draft, mirroring skill_grouper.py's
    pattern for the same situation.
    """
    print("\n=== Resume summary ===")

    if not drafted_summary.strip():
        print("  [Notice] LLM did not produce a usable summary.")
        while True:
            raw = input("  Enter a summary manually ('o <text>'), or 'skip' to omit entirely: ").strip()
            decision = parse_review_input(raw)
            if decision.action == "override":
                return decision.value
            if decision.action == "skip":
                return ""
            print(f"  [Error] {decision.value}")

    current_draft = drafted_summary

    while True:
        print(f"  Drafted summary: {current_draft}")
        raw = input(
            "\n  Accept this summary? (Enter to accept / 'r' or 'r <text>' to regenerate / "
            "'o <text>' to override / 'skip' to omit entirely): "
        ).strip()
        decision = parse_review_input(raw)

        if decision.action == "regenerate":
            if job_description is None or final_bullets is None or education_entries is None:
                print(
                    "  [Warning] Cannot regenerate -- this summary was confirmed without full "
                    "context. Continuing with the current draft."
                )
                continue
            print("  Regenerating...")
            current_draft = draft_summary(
                job_description, final_bullets, education_entries,
                domain_job_function_tags, steering_note=decision.value,
            )
            continue

        if decision.action == "accept":
            print("  -> Using drafted summary")
            return current_draft

        if decision.action == "override":
            print("  -> Using manually entered text")
            return decision.value

        if decision.action == "skip":
            print("  -> Summary omitted from this resume")
            return ""

        print(f"  [Error] {decision.value}")


def draft_and_confirm_summary(job_description, final_bullets, education_entries,
                               domain_job_function_tags=None):
    """Main entry point: single LLM draft + mandatory human confirmation
    (including the (v3) regenerate option). Returns the final summary
    text, or None if the user chose to omit it entirely.
    """
    drafted = draft_summary(job_description, final_bullets, education_entries, domain_job_function_tags)
    confirmed = confirm_summary(
        drafted,
        job_description=job_description,
        final_bullets=final_bullets,
        education_entries=education_entries,
        domain_job_function_tags=domain_job_function_tags,
    )
    return confirmed if confirmed else None
