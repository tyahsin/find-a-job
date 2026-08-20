"""
track/job_extractor.py
Version: v3 (2026-08-20)

Extracts the 8 job fields (matching core/db.py's JobORM / the `job`
table in 001_init_schema.sql) from a raw JD text blob, via a single
batch LLM call. Output is meant to be reviewed by the user before being
passed into track/job_app_tracker.py's find_or_create_job() -- this
module does NOT write to the database itself; it only produces a dict
of candidate field values. (Consumed by track/cli.py's `apply create
--jd-text`/`--jd-file` path -- the standalone `apply extract` command
was removed in cli.py v2.)

Field scope (user-confirmed 2026-08-20, no additions/removals from what
application_logger.py's find_or_create_job() already accepts):
    company_name, job_title, job_location, job_description,
    source_url, date_posted, contact_name, contact_email

(v2 fix, 2026-08-20): v1 incorrectly imported a non-existent
`call_llm()` function from core/llm_client.py. Fixed to use
get_llm_client().generate_text(...).text.

(v3 fix, 2026-08-20): real-world test surfaced that some LLM responses
wrap the JSON payload in a markdown code fence (```json ... ```) even
though _SYSTEM_PROMPT explicitly asks for raw JSON only -- prompt
wording alone does not reliably prevent this across models/providers.
Added _strip_code_fence() to defensively strip a leading/trailing
```json or ``` fence (with optional surrounding whitespace) before
attempting json.loads(). This is a parsing-robustness fix only; it
does not change the function signature, the field scope, or any
governance behavior (still: extract literally, never invent values).

Design principles carried over from the rest of this codebase:
- LLM only extracts/reformats what's literally present in the JD text;
  it must not invent a company name, contact, or date that isn't in
  the source text. Fields not found in the JD are returned as None
  rather than guessed.
- job_description is the one field that should normally just be the
  JD text itself (or a lightly cleaned version of it) -- NOT a summary.
- date_posted parsing: the LLM returns its best-guess raw string; a
  local parser then normalizes to a date object, returning None on any
  ambiguity rather than guessing.
  IMPORTANT for callers: this function returns date_posted as an
  actual date object (or None), already parsed -- do NOT re-parse it
  as a string in apply/cli.py; that conversion only applies to the
  manual/no-LLM CLI args, which are raw strings from the command line.
- Low temperature (0.0) for this call: field extraction should be as
  literal/deterministic as possible.
- No confirmation gate is implemented inside this module itself -- the
  human-confirmation step happens one layer up, in track/cli.py's
  `apply create --jd-text`/`--jd-file` path.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Optional

from core.llm_client import get_llm_client

JOB_FIELDS = (
    "company_name",
    "job_title",
    "job_location",
    "job_description",
    "source_url",
    "date_posted",
    "contact_name",
    "contact_email",
)

_SYSTEM_PROMPT = """You are extracting structured fields from a job description (JD) posting. Extract ONLY information that is literally present in the JD text provided by the user. Do not invent, infer, or guess a value for any field that is not explicitly stated in the text. Return ONLY a JSON object, no other text, no markdown code fences."""

_USER_PROMPT_TEMPLATE = """Return a JSON object with exactly these keys:
- company_name (string or null)
- job_title (string or null)
- job_location (string or null)
- job_description (string): the full JD text, lightly cleaned of obvious webpage boilerplate (e.g. "Apply now" buttons, nav menus) but otherwise preserved in full -- this is NOT a summary.
- date_posted (string or null): return the raw date phrase exactly as written in the text (e.g. "Posted 3 days ago", "2026-08-15"); do not convert or calculate it yourself.
- contact_name (string or null)
- contact_email (string or null)

JD text:
---
{jd_text}
---"""

_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL
)


def _strip_code_fence(raw_text: str) -> str:
    """Strips a leading/trailing markdown code fence (```json ... ``` or
    ``` ... ```) from an LLM response, if present. Some models wrap JSON
    output in a fence even when explicitly told not to (prompt wording
    alone does not reliably prevent this) -- this is a defensive parsing
    step, not a change to what content is trusted/extracted.

    If no fence is detected, returns the input unchanged (already-raw
    JSON responses continue to work exactly as before).
    """
    match = _CODE_FENCE_PATTERN.match(raw_text.strip())
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def _parse_date_posted(raw_value: Optional[str]) -> Optional[date]:
    if not raw_value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw_value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def extract_job_fields(jd_text: str, source_url: Optional[str] = None) -> dict:
    """Extracts the 8 job fields from raw JD text via a single LLM call.

    source_url is passed in directly (not extracted by the LLM). Returns
    a dict with all 8 JOB_FIELDS keys populated (None where not found/
    not parseable) -- date_posted is already a date object (or None).
    Does not write to the database.
    """
    user_prompt = _USER_PROMPT_TEMPLATE.format(jd_text=jd_text)

    llm_client = get_llm_client()
    response = llm_client.generate_text(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=4000,
    )
    raw_text = response.text
    cleaned_text = _strip_code_fence(raw_text)

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM did not return valid JSON for job field extraction: {exc}\n"
            f"Raw response: {raw_text!r}"
        ) from exc

    result = {field: parsed.get(field) for field in JOB_FIELDS}
    result["source_url"] = source_url if source_url is not None else parsed.get("source_url")
    result["date_posted"] = _parse_date_posted(result.get("date_posted"))

    return result
