"""
gencv/orphan_selector.py

Shared relevance-matching for "orphan" records -- education degrees or work
positions that have no claims selected for a given JD, but might still be
worth showing on the resume. All three orphan-judging flows share one
batch-LLM skeleton (judge_orphans); only the per-record formatting and the
system prompt differ.
"""

from __future__ import annotations

import re
from typing import Callable, TypeVar

from core.db import DegreeORM, PositionORM
from core.llm_client import get_llm_client

T = TypeVar("T")  # DegreeORM or PositionORM


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ORPHAN_DEGREE_SYSTEM_PROMPT = """You are a relevance-matching assistant. You will be given a job description
and a list of degree/education records (each with an index, degree level,
degree type, subject area, and description).

Judge which of these degrees are relevant enough to include in a resume
tailored to this job -- based on whether the subject area, degree level, or
description plausibly relates to the job's requirements or domain.

Output format: output the index numbers of all relevant degrees, one per
line, no other text, no explanation. If none are relevant, output NONE
instead."""

ORPHAN_POSITION_SYSTEM_PROMPT = """You are a relevance-matching assistant. You will be given a job description
and a list of work experience records (each with an index, job title,
department, and responsibility description).

Judge which of these positions are relevant enough to include in a resume
tailored to this job -- based on whether the role, department, or
responsibilities plausibly relate to the job's requirements or domain.

Output format: output the index numbers of all relevant positions, one per
line, no other text, no explanation. If none are relevant, output NONE
instead."""

_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# {record_noun} with no supporting claims ({count} total)
{records_block}

Output the index numbers of all relevant records:"""


# ---------------------------------------------------------------------------
# Per-record formatters -- the only content that differs between record types
# ---------------------------------------------------------------------------

def _format_degree(degree: DegreeORM) -> str:
    return (
        f"Level: {degree.degree_level or 'not specified'}\n"
        f"    Type: {degree.degree_type or 'not specified'}\n"
        f"    Subject: {degree.degree_name or 'not specified'}\n"
        f"    Description: {degree.degree_description or 'not specified'}"
    )


def _format_position(position: PositionORM) -> str:
    return (
        f"Title: {position.title or 'not specified'}\n"
        f"    Department: {position.department or 'not specified'}\n"
        f"    Responsibility: {position.responsibility or 'not specified'}"
    )


# ---------------------------------------------------------------------------
# Shared skeleton
# ---------------------------------------------------------------------------

def _parse_relevant_indices(raw_response: str) -> set[int]:
    """Tolerant parsing of a newline-separated list of index numbers."""
    if "NONE" in raw_response.upper() and not re.search(r"\d", raw_response):
        return set()
    return {int(n) for n in re.findall(r"\d+", raw_response)}


def judge_orphans(
    orphans: list[T],
    job_description: str,
    system_prompt: str,
    record_formatter: Callable[[T], str],
    record_noun: str,
) -> list[T]:
    """Single batch LLM call over orphan records only. Skipped entirely
    (zero LLM calls) if orphans is empty. Returns the SUBSET judged relevant,
    same ORM type as the input.

    record_formatter turns ONE ORM row into its indented text block (without
    the leading '[i] ' index prefix -- this function adds that). record_noun
    is a short human label for the batch, e.g. 'Degree records' or
    'Work experience records'."""
    if not orphans:
        return []

    index_to_record: dict[int, T] = {}
    lines = []
    for i, record in enumerate(orphans, 1):
        index_to_record[i] = record
        lines.append(f"[{i}] {record_formatter(record)}")

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        record_noun=record_noun,
        count=len(orphans),
        records_block="\n\n".join(lines),
    )

    response = get_llm_client().generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=200,
    )

    relevant_indices = _parse_relevant_indices(response.text)
    return [index_to_record[i] for i in relevant_indices if i in index_to_record]


# ---------------------------------------------------------------------------
# Thin public wrappers -- keep call sites readable and type-specific
# ---------------------------------------------------------------------------

def judge_orphan_degrees(orphan_degrees: list[DegreeORM], job_description: str) -> list[DegreeORM]:
    return judge_orphans(
        orphan_degrees, job_description,
        system_prompt=ORPHAN_DEGREE_SYSTEM_PROMPT,
        record_formatter=_format_degree,
        record_noun="Degree records",
    )


def judge_orphan_positions(orphan_positions: list[PositionORM], job_description: str) -> list[PositionORM]:
    return judge_orphans(
        orphan_positions, job_description,
        system_prompt=ORPHAN_POSITION_SYSTEM_PROMPT,
        record_formatter=_format_position,
        record_noun="Work experience records",
    )