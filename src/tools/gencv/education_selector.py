"""
education_selector.py
Version: v1 (2026-08-15)

Determines which degrees should appear in the final resume's education
section. Two independent paths, unioned together:

Path A: degrees that already have at least one claim in select_claims()'s
        selected_claim_ids -- treated as "relevance already established" by
        that earlier batch LLM judgment. No additional LLM call needed here.

Path B: "orphan" degrees -- degrees with ZERO claims attached at all (never
        entered select_claims()'s candidate pool in the first place, since
        that pool is sourced from ClaimORM). These need a SEPARATE batch LLM
        judgment call, using the degree's own static fields
        (degree_level/degree_type/degree_name/degree_description) instead
        of claim content. If there are no orphan degrees, this call is
        skipped entirely -- zero additional cost in that case.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

from __future__ import annotations

import re
import uuid

from core.db import get_session, DegreeORM, ClaimORM, ActionORM, ProblemORM, ProjectORM
from core.llm_client import get_llm_client

ORPHAN_DEGREE_SYSTEM_PROMPT = """You are a relevance-matching assistant. You will be given a job description
and a list of degree/education records (each with an index, degree level,
degree type, subject area, and description).

Judge which of these degrees are relevant enough to include in a resume
tailored to this job -- based on whether the subject area, degree level, or
description plausibly relates to the job's requirements or domain.

Output format: output the index numbers of all relevant degrees, one per
line, no other text, no explanation. If none are relevant, output NONE
instead."""

ORPHAN_DEGREE_USER_PROMPT_TEMPLATE = """# Job Description
{job_description}

# Degree records with no supporting claims ({count} total)
{degrees_block}

Output the index numbers of all relevant degrees:"""


def _fetch_claim_backed_degree_ids(session, claim_ids: list[str]) -> set[str]:
    """Path A: returns the set of degree_id (as strings) for every claim in
    claim_ids that traces back to an education-track project (i.e.
    project.degree_id is not null)."""
    if not claim_ids:
        return set()

    uuid_ids = [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]

    rows = (
        session.query(ProjectORM.degree_id)
        .join(ProblemORM, ProblemORM.project_id == ProjectORM.project_id)
        .join(ActionORM, ActionORM.problem_id == ProblemORM.problem_id)
        .join(ClaimORM, ClaimORM.action_id == ActionORM.action_id)
        .filter(ClaimORM.claim_id.in_(uuid_ids))
        .filter(ProjectORM.degree_id.isnot(None))
        .distinct()
        .all()
    )
    return {str(row[0]) for row in rows}


def _fetch_orphan_degrees(session, claim_backed_degree_ids: set[str]) -> list[DegreeORM]:
    """Path B candidates: every DegreeORM row whose degree_id is NOT in
    claim_backed_degree_ids."""
    all_degrees = session.query(DegreeORM).all()
    backed_ids = {uuid.UUID(d) if isinstance(d, str) else d for d in claim_backed_degree_ids}
    return [d for d in all_degrees if d.degree_id not in backed_ids]


def _build_orphan_degree_prompt(
    orphan_degrees: list[DegreeORM], job_description: str
) -> tuple[str, dict[int, DegreeORM]]:
    """Builds a single batch prompt listing every orphan degree by index."""
    index_to_degree: dict[int, DegreeORM] = {}
    lines = []
    for i, degree in enumerate(orphan_degrees, 1):
        index_to_degree[i] = degree
        lines.append(
            f"[{i}] Level: {degree.degree_level or 'not specified'}\n"
            f"    Type: {degree.degree_type or 'not specified'}\n"
            f"    Subject: {degree.degree_name or 'not specified'}\n"
            f"    Description: {degree.degree_description or 'not specified'}"
        )

    degrees_block = "\n\n".join(lines)
    user_prompt = ORPHAN_DEGREE_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        count=len(orphan_degrees),
        degrees_block=degrees_block,
    )
    return user_prompt, index_to_degree


def _parse_relevant_indices(raw_response: str) -> set[int]:
    """Tolerant parsing of a newline-separated list of index numbers."""
    if "NONE" in raw_response.upper() and not re.search(r"\d", raw_response):
        return set()
    return {int(n) for n in re.findall(r"\d+", raw_response)}


def _judge_orphan_degrees(
    orphan_degrees: list[DegreeORM], job_description: str
) -> list[DegreeORM]:
    """Single batch LLM call over orphan degrees only. Skipped entirely
    (zero LLM calls) if orphan_degrees is empty."""
    if not orphan_degrees:
        return []

    user_prompt, index_to_degree = _build_orphan_degree_prompt(orphan_degrees, job_description)

    response = get_llm_client().generate_text(
        system_prompt=ORPHAN_DEGREE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=200,
    )

    relevant_indices = _parse_relevant_indices(response.text)
    return [index_to_degree[i] for i in relevant_indices if i in index_to_degree]


def select_relevant_degrees(session, selected_claim_ids: list[str], job_description: str) -> list[DegreeORM]:
    """Main entry point. Returns the union of Path A and Path B degrees.
    No mandatory human confirmation gate -- lower stakes than identity/
    network selection."""
    claim_backed_degree_ids = _fetch_claim_backed_degree_ids(session, selected_claim_ids)

    path_a_degrees = (
        session.query(DegreeORM)
        .filter(DegreeORM.degree_id.in_(
            uuid.UUID(d) if isinstance(d, str) else d for d in claim_backed_degree_ids
        ))
        .all()
        if claim_backed_degree_ids
        else []
    )

    orphan_degrees = _fetch_orphan_degrees(session, claim_backed_degree_ids)
    path_b_degrees = _judge_orphan_degrees(orphan_degrees, job_description)

    seen_ids: set = set()
    result: list[DegreeORM] = []
    for degree in path_a_degrees + path_b_degrees:
        if degree.degree_id not in seen_ids:
            seen_ids.add(degree.degree_id)
            result.append(degree)

    return result
