"""
track/prompts.py
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Optional
from collections.abc import Iterable

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

    if isinstance(result, Iterable): # result can't be NoneType (i.e. no data parsed successfully)
        raise TypeError("Contents failed to be scraped from webpage URL. Please paste the context instead.")

    return result
