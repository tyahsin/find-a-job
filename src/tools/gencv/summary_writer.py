"""
summary_writer.py
Version: v1 (2026-08-15)

Drafts a short professional summary (2-3 sentences, hard cap: 3) that sits
right after personal information and before all other sections in the
final resume -- a brief overview of the candidate + the job description,
bridging any gap between them ONLY IF ONE ACTUALLY EXISTS.

Design decisions (2026-08-15):
- Single LLM call (not batch) -- there is only one summary to draft per
  generate_cv() run.
- MANDATORY human confirmation gate -- genuinely new generated text
  summarizing the entire resume. Same safety pattern as
  experience_rewriter.py's _confirm_one(): Enter to accept / type new
  text to override.
- Input is deliberately restricted to CONFIRMED content only -- the final
  fused STAR bullets and the selected education entries -- never the full
  claim database.
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
section), and a list of the candidate's education entries. Your task is
to write a short professional summary that will appear at the top of the
resume, right after the candidate's contact information.

Strict rules (must follow):
1. Base the summary ONLY on the bullet points and education entries given
   to you -- do not reference any experience, skill, or achievement that
   is not reflected in this material.
2. Do not add any concrete fact, number, or degree of achievement beyond
   what the bullet points already state.
3. Judge whether there is an actual, meaningful gap between the
   candidate's background (as shown in the bullets/education) and the job
   description's requirements.
   - If a real gap exists, briefly acknowledge the transition/bridge in a
     factual, non-defensive way (e.g. framing transferable skills), without
     overstating readiness.
   - If NO meaningful gap exists -- i.e. the candidate's background already
     aligns well with the job -- do NOT invent or imply a gap. Simply
     summarize the candidate's relevant strengths. Manufacturing a gap
     narrative where none exists is not acceptable.
4. Write in active voice, third person is not used (do not refer to the
   candidate as "he/she/they" or by name) -- write as a direct, resume-
   style summary (e.g. "Backend engineer with 5 years of experience
   building...", not "This candidate has...").
5. HARD LIMIT: the summary must be AT MOST 3 sentences. Prefer 2 sentences
   when 2 are sufficient -- do not pad to reach 3.
6. No filler words, no generic claims not backed by the given material
   (e.g. avoid empty phrases like "highly motivated" or "team player"
   unless a bullet point actually demonstrates it).
7. Output must be in English, regardless of the input language.

Output format: output ONLY the summary text itself, plain text, no
preamble, no heading, no explanation, no quotation marks."""

SUMMARY_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Confirmed resume bullet points ({bullet_count} total)
{bullets_block}

# Education entries ({education_count} total)
{education_block}

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


def draft_summary(
    job_description: str,
    final_bullets: list[str],
    education_entries: list[dict],
) -> str:
    """Single LLM call. Returns the drafted summary text (not yet
    confirmed -- see confirm_summary() for the mandatory human gate)."""
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        bullet_count=len(final_bullets),
        bullets_block=_format_bullets_block(final_bullets),
        education_count=len(education_entries),
        education_block=_format_education_block(education_entries),
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
) -> str | None:
    """Main entry point: single LLM draft + mandatory human confirmation.
    Returns the final summary text, or None if the user chose to omit it
    entirely."""
    drafted = draft_summary(job_description, final_bullets, education_entries)
    confirmed = confirm_summary(drafted)
    return confirmed if confirmed else None
