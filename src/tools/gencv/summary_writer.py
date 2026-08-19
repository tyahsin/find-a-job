"""
summary_writer.py
Version: v2 (2026-08-16)

Changelog:
- v2 (2026-08-16): Added domain/job_function tags as an auxiliary signal
input. Rationale (per user decision, 2026-08-16): domain (industry
knowledge) and job_function (role-type) tags are being excluded entirely
from the resume's skills section (see resume_builder.py v13's
build_skills_section(), which now filters to tag_category == "skill"
only) -- these two tag categories are conceptually about WHAT DOMAIN/ROLE
the candidate operates in, not a technical skill list, and are a better
fit for the summary/headline framing than a skills bullet list.
draft_summary() / draft_and_confirm_summary() now accept an optional
domain_job_function_tags parameter. Per explicit user confirmation, these
tags are AUXILIARY CONTEXT ONLY -- the system prompt instructs the LLM to
use them as background signal for word choice/framing, NOT to force their
literal appearance in the summary text. This preserves the existing
"no forced keyword stuffing" principle and the hard 3-sentence cap: a
forced requirement to mention every domain/job_function tag would compete
with bullet-point content for very limited space.
- v1 (2026-08-15): Initial draft_summary()/draft_and_confirm_summary(),
mandatory human confirmation gate, "bridge the gap only if real" logic.

Drafts a short professional summary (2-3 sentences, hard cap: 3) that sits
right after personal information and before all other sections in the
final resume -- a brief overview of the candidate + the job description,
bridging any gap between them ONLY IF ONE ACTUALLY EXISTS.

Design decisions:
- Single LLM call (not batch) -- there is only one summary to draft per
generate_cv() run.
- MANDATORY human confirmation gate -- genuinely new generated text
summarizing the entire resume. Same safety pattern as
experience_rewriter.py's _confirm_one(): Enter to accept / type new
text to override.
- Input is deliberately restricted to CONFIRMED content only -- the final
fused STAR bullets, the selected education entries, and (v2) the
domain/job_function tags attached to the resume's included claims --
never the full claim database.
- The "bridge the gap" instruction is explicitly conditional: if the
candidate's confirmed experience already aligns well with the job
description, the summary should NOT manufacture a gap-bridging
narrative -- inventing a gap where none exists would itself be a
fabrication.
- Runs AFTER rewrite_and_confirm() (needs final confirmed bullet text) and
AFTER education selection, but BEFORE build_resume_yaml().

English-only prompt and output, per the project-wide LANGUAGE POLICY.
"""

from __future__ import annotations

from core.llm_client import get_llm_client

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
use them to inform word choice and framing (e.g. knowing the candidate
has "Fintech" domain experience might shape how you phrase a bullet
reference), but you are NOT required to literally mention every tag,
or any tag at all, in the summary text. Do not force tag keywords into
the summary just because they were provided -- only use them if they
naturally strengthen the summary within the length limit.
4. Judge whether there is an actual, meaningful gap between the
candidate's background (as shown in the bullets/education) and the job
description's requirements.
- If a real gap exists, briefly acknowledge the transition/bridge in a
factual, non-defensive way (e.g. framing transferable skills), without
overstating readiness.
- If NO meaningful gap exists -- i.e. the candidate's background already
aligns well with the job -- do NOT invent or imply a gap. Simply
summarize the candidate's relevant strengths. Manufacturing a gap
narrative where none exists is not acceptable.
5. Write in active voice, third person is not used (do not refer to the
candidate as "he/she/they" or by name) -- write as a direct, resume-
style summary (e.g. "Backend engineer with 5 years of experience
building...", not "This candidate has...").
6. HARD LIMIT: the summary must be AT MOST 3 sentences. Prefer 2 sentences
when 2 are sufficient -- do not pad to reach 3.
7. No filler words, no generic claims not backed by the given material
(e.g. avoid empty phrases like "highly motivated" or "team player"
unless a bullet point actually demonstrates it).
8. Output must be in English, regardless of the input language.

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

Write the professional summary (at most 3 sentences):"""


def _format_bullets_block(final_bullets: list[str]) -> str:
    if not final_bullets:
        return "(no experience bullets in this resume)"
    return "\n".join(f"- {b}" for b in final_bullets)


def _format_education_block(education_entries: list[dict]) -> str:
    if not education_entries:
        return "(no education entries in this resume)"
    lines = []
    for e in education_entries:
        parts = [e.get("institution", ""), e.get("degree", ""), e.get("area", "")]
        lines.append("- " + ", ".join(p for p in parts if p))
    return "\n".join(lines)


def _format_tags_block(domain_job_function_tags: list[str] | None) -> str:
    """(v2) domain/job_function tag 清單的呈現格式。空清單時明確標示無資料，
    而非留白讓 LLM 自行猜測。"""
    if not domain_job_function_tags:
        return "(no domain/job-function tags available for this resume)"
    return ", ".join(sorted(set(domain_job_function_tags)))


def draft_summary(
    job_description: str,
    final_bullets: list[str],
    education_entries: list[dict],
    domain_job_function_tags: list[str] | None = None,
) -> str:
    """Single LLM call. Returns the drafted summary text (not yet
    confirmed -- see confirm_summary() for the mandatory human gate).

    (v2) domain_job_function_tags is optional auxiliary context -- see
    module docstring and SUMMARY_SYSTEM_PROMPT rule 3 for why these tags
    are advisory only, never a forced-inclusion requirement.
    """
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        bullet_count=len(final_bullets),
        bullets_block=_format_bullets_block(final_bullets),
        education_count=len(education_entries),
        education_block=_format_education_block(education_entries),
        tag_count=len(domain_job_function_tags) if domain_job_function_tags else 0,
        tags_block=_format_tags_block(domain_job_function_tags),
    )

    response = get_llm_client().generate_text(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=200,
    )

    return response.text.strip()


def confirm_summary(drafted_summary: str) -> str:
    """Mandatory human confirmation gate -- mirrors
    experience_rewriter._confirm_one() exactly."""
    print("\n=== Resume summary ===")

    if not drafted_summary.strip():
        print("  [Notice] LLM did not produce a usable summary.")
        choice = input("  Enter a summary manually (or leave blank to skip entirely): ").strip()
        return choice or ""

    print(f"  Drafted summary: {drafted_summary}")
    choice = input("\n  Accept this summary? (Enter to accept / type new text to override / type 'skip' to omit entirely): ").strip()

    if choice.lower() == "skip":
        print("  -> Summary omitted from this resume")
        return ""
    if choice:
        print("  -> Using manually entered text")
        return choice
    print("  -> Using drafted summary")
    return drafted_summary


def draft_and_confirm_summary(
    job_description: str,
    final_bullets: list[str],
    education_entries: list[dict],
    domain_job_function_tags: list[str] | None = None,
) -> str | None:
    """Main entry point: single LLM draft + mandatory human confirmation.
    Returns the final summary text, or None if the user chose to omit it
    entirely.

    (v2) domain_job_function_tags: optional auxiliary tag list, passed
    through to draft_summary(). See module docstring for rationale.
    """
    drafted = draft_summary(job_description, final_bullets, education_entries, domain_job_function_tags)
    confirmed = confirm_summary(drafted)
    return confirmed if confirmed else None
