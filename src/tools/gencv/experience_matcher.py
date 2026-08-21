"""
experience_matcher.py
Version: v4 (2026-08-16)

Changelog:
- v4 (2026-08-16): LANGUAGE POLICY fix -- RELEVANCE_SYSTEM_PROMPT and
_build_batch_prompt()'s internal labels (background/problem/action/
result/tags/operating notes, "none", "no background info") were still
in Chinese, inconsistent with the rest of the gencv pipeline which is
English-only end-to-end (education_selector.py, experience_rewriter.py,
summary_writer.py, cli.py already were). No logic, function signatures,
or control flow changed -- this is a pure string-content translation.
- v3 (2026-08-16): Added degree.degree_id -> context["degree_id"] to
_EXPERIENCE_CONTEXT_SQL / fetch_experience_contexts(). This was missing
from v2 -- resume_builder.build_project_entries() needs each
education-track claim's degree_id to collect degree_ids_with_projects
(the set build_education_entries() uses to filter relevant_degrees down
to only degrees with at least one selected project underneath them).
Without this column, ctx.get("degree_id") was always None and the
education filtering logic in resume_builder.py silently produced an
empty education section every time. Also added project.project_id ->
context["project_id"] for symmetry (not currently consumed elsewhere,
but the query already joins through project).
NOTE (v12 of resume_builder.py, 2026-08-16): the degree_ids_with_projects
mechanism this comment refers to has since been REMOVED from
resume_builder.py -- build_education_entries() now uses relevant_degrees
directly with no post-filtering. context["degree_id"] is therefore no
longer consumed by any caller as of resume_builder.py v12. Left in place
(harmless, zero cost) in case a future feature needs per-claim degree
linkage again; flagged here so this doesn't get mistaken for dead code
with no explanation.
- v2 (2026-08-12): Fixed two gaps between this file's actual behavior and the
intended design recorded in project notes:
1. Governance filtering (_resolve_candidate_ids, replaces the old
_allowed_governance): previously only claims WITH a governance row AND
allowed_for_generation = True were kept, silently excluding every claim
that predates the governance layer (no governance row at all). Now the
candidate pool = all claims, MINUS claims with an explicit
allowed_for_generation = False. Claims with no governance row default to
allowed, matching the schema's allowed_for_generation BOOLEAN DEFAULT
TRUE design intent.
2. Tag signal (fetch_tags_for_claims, new): the batch relevance prompt now
includes each candidate's claim_tag-linked tags (skill/domain/
job_function), with an explicit system-prompt instruction that tags are
an important relevance signal. Previously the LLM only saw free-text
problem/action/result descriptions and had no visibility into
user-confirmed tags.
- v1 (2026-08-10, inferred): Initial batch-LLM relevance matcher.

JD -> determines which past experiences are relevant to this job posting,
producing the candidate shortlist for this resume generation run.
Shared query utility: fetches, in one pass, the full background chain
claim -> action -> problem -> project -> position/company (or degree/school).

core/db.py's ORM models only declare Column(ForeignKey(...)) constraints,
with no relationship() defined, so attribute-chain access like
claim.action.problem... is not available -- explicit JOIN queries are
required.

Flow (computationally light design, sized for a candidate pool of
~250-300 rows):
Step 1: Governance-layer filtering -- claims with no governance row default
to allowed (matching the schema's allowed_for_generation BOOLEAN DEFAULT
TRUE design intent); only claims with an explicit
allowed_for_generation = False are excluded.
Step 2: Single batch LLM call -- all candidate experience summaries
(including tag labels) are packed into one prompt, one LLM call, asking
it to return the relevant claim_id list directly. This replaces a
"one LLM call per experience" approach; if the candidate pool ever grows
well past 1000+ rows, consider adding a vector-similarity pre-filter
(using claim_chunks embeddings) before this step.

English-only prompts and output, per the project-wide LANGUAGE POLICY.
"""

import re
import uuid
from collections import defaultdict

from sqlalchemy import text
from core.db import get_session, GovernanceORM, ClaimORM, ClaimTagORM, TagORM
from core.llm_client import get_llm_client

RELEVANCE_SYSTEM_PROMPT = """You are a resume screening assistant. You will be given a job description
and a list of the candidate's past experience records (each with an index,
background, problem, action, result, tags, and that record's usage policy).


Judge whether each experience record is relevant to this job posting and
worth including in a resume tailored to it.


A record is relevant if EITHER of the following is true:
- The business domain or work setting matches the job description (e.g.
  same industry, same type of stakeholder collaboration), OR
- The technical skills, methods, or tools demonstrated in the record match
  skills explicitly required or mentioned in the job description — even if
  the project's domain or setting (e.g. academic project vs. corporate
  role) differs from the job posting's context.


Do not require both domain match AND skill match simultaneously. A
technically strong record demonstrating a required skill (e.g. a specific
ML method, a specific tool, hands-on model development) should be marked
relevant even if it comes from an academic or personal project, as long as
the skill itself is explicitly called for in the job description.


Tags are an important relevance signal -- tags represent skills/domains/job
functions the user has manually confirmed for that record. Even if the
result text itself has no obvious keyword match, a tag that matches the
job's requirements should still be treated as relevant.


If a record's usage policy (use_policy) explicitly excludes this type of
job, always treat it as not relevant, regardless of tag or content match.

Output format: output only the index numbers of relevant records, one per
line, no other text, no explanation beyond the numbers. If none are
relevant, output NONE."""

_EXPERIENCE_CONTEXT_SQL = """
SELECT
    claim.claim_id,
    claim.claim_description AS claim_description,
    action.action_description AS action_description,
    action.role_scope,
    problem.problem_description AS problem_description,
    problem.stakeholder_impact,
    project.project_id,
    project.project_name,
    project.project_context,
    position.title AS position_title,
    position.department,
    position.start_date AS position_start,
    position.end_date AS position_end,
    company.company_name,
    degree.degree_id,
    degree.degree_level,
    degree.degree_name,
    degree.start_date AS degree_start,
    degree.end_date AS degree_end,
    school.school_name
FROM claim
JOIN action ON claim.action_id = action.action_id
JOIN problem ON action.problem_id = problem.problem_id
JOIN project ON problem.project_id = project.project_id
LEFT JOIN position ON project.position_id = position.position_id
LEFT JOIN company ON position.company_id = company.company_id
LEFT JOIN degree ON project.degree_id = degree.degree_id
LEFT JOIN school ON degree.school_id = school.school_id
WHERE claim.claim_id = ANY(:claim_ids)
"""

def to_uuid_list(claim_ids: list) -> list[uuid.UUID]:
    """claim_id is always passed externally as a string (UUID text); convert
    to uuid.UUID objects before filtering against a SQLAlchemy PGUUID column."""
    return [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]

def fetch_experience_contexts(session, claim_ids: list[str]) -> dict[str, dict]:
    """Returns a {claim_id string: context_dict} mapping, fetched in one JOIN query.

    is_work_track=True means this record hangs off a position (work
    experience); False means it hangs off a degree (education experience) --
    project.position_id / degree_id are mutually exclusive.

    degree_id (added in v3): when is_work_track=False, this claim's degree_id
    (as a string). Originally consumed by resume_builder.build_project_entries()
    to collect degree_ids_with_projects for build_education_entries() to filter
    on -- that filtering mechanism was removed in resume_builder.py v12, so this
    field currently has no consumer, but is kept for potential future use.
    None when is_work_track=True.
    """
    if not claim_ids:
        return {}

    rows = session.execute(
        text(_EXPERIENCE_CONTEXT_SQL), {"claim_ids": to_uuid_list(claim_ids)}
    ).mappings().all()

    contexts = {}
    for row in rows:
        claim_id = str(row["claim_id"])
        is_work_track = row["position_title"] is not None
        contexts[claim_id] = {
            "claim_description": row["claim_description"],
            "action_description": row["action_description"],
            "role_scope": row["role_scope"],
            "problem_description": row["problem_description"],
            "stakeholder_impact": row["stakeholder_impact"],
            "project_id": str(row["project_id"]) if row["project_id"] else None,
            "project_name": row["project_name"],
            "project_context": row["project_context"],
            "is_work_track": is_work_track,
            "org_name": row["company_name"] if is_work_track else row["school_name"],
            "title": row["position_title"] if is_work_track else row["degree_name"],
            "start_date": row["position_start"] if is_work_track else row["degree_start"],
            "end_date": row["position_end"] if is_work_track else row["degree_end"],
            "degree_id": str(row["degree_id"]) if (not is_work_track and row["degree_id"]) else None,
        }
    return contexts

def fetch_tags_for_claims(session, claim_ids: list[str]) -> dict[str, list[str]]:
    """claim_id -> [tag_name, ...] mapping, fetched in one JOIN query covering
    every claim's tags.

    (v2) Used to feed the user's manually-confirmed tags into the batch
    relevance prompt, as a clearer relevance signal than free-text result
    descriptions alone.
    """
    if not claim_ids:
        return {}

    rows = (
        session.query(ClaimTagORM.claim_id, TagORM.tag_name)
        .join(TagORM, ClaimTagORM.tag_id == TagORM.tag_id)
        .filter(ClaimTagORM.claim_id.in_(to_uuid_list(claim_ids)))
        .all()
    )
    tags_map: dict[str, list[str]] = defaultdict(list)
    for claim_id, tag_name in rows:
        tags_map[str(claim_id)].append(tag_name)
    return tags_map

def _resolve_candidate_ids(session) -> tuple[list[str], dict[str, GovernanceORM]]:
    """Step 1: governance-layer filtering.

    Candidate pool = all claim_ids, minus any claim that has a governance row
    with allowed_for_generation explicitly set to False. Claims with no
    governance row at all (e.g. legacy claims created before the governance
    layer existed) default to allowed, matching the schema's
    allowed_for_generation BOOLEAN DEFAULT TRUE design intent.

    Returns:
        candidate_ids: claim_id strings allowed into the candidate pool.
        governance_map: claim_id -> GovernanceORM, containing only claims that
        actually have a governance row (used later to read use_policy/
        notes_for_llm when building the prompt; claims absent from this map
        should be treated by the caller as having no governance record).
    """
    all_claim_ids = [str(row[0]) for row in session.query(ClaimORM.claim_id).all()]

    governance_rows = session.query(GovernanceORM).all()
    governance_map = {str(g.claim_id): g for g in governance_rows}

    candidate_ids = [
        cid for cid in all_claim_ids
        if governance_map.get(cid) is None or governance_map[cid].allowed_for_generation is not False
    ]
    return candidate_ids, governance_map

def _build_batch_prompt(
    candidate_ids: list[str],
    contexts: dict[str, dict],
    governances: dict[str, GovernanceORM],
    tags_map: dict[str, list[str]],
    job_description: str,
) -> tuple[str, dict[int, str]]:
    """Builds a single batch prompt listing every candidate experience by
    index (including tags). Returns (prompt, {index: claim_id})."""
    index_to_claim_id: dict[int, str] = {}
    lines = []
    for i, claim_id in enumerate(candidate_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        index_to_claim_id[i] = claim_id

        governance = governances.get(claim_id)
        use_policy = governance.use_policy if governance else None
        notes_for_llm = governance.notes_for_llm if governance else None

        tags = tags_map.get(claim_id, [])
        tags_line = ", ".join(tags) if tags else "none"

        org_context = f"{ctx['org_name']} - {ctx['title']}" if ctx["org_name"] else "no background info"
        lines.append(
            f"[{i}] Background: {org_context}\n"
            f"    Problem: {ctx['problem_description']}\n"
            f"    Action: {ctx['action_description']}\n"
            f"    Result: {ctx['claim_description']}\n"
            f"    Tags: {tags_line}\n"
            f"    Usage policy: use_policy={use_policy or 'none'}, "
            f"notes_for_llm={notes_for_llm or 'none'}"
        )

    experiences_block = "\n\n".join(lines)
    user_prompt = f"""# Job Description
{job_description}

# Past experience records ({len(index_to_claim_id)} total)
{experiences_block}"""

    return user_prompt, index_to_claim_id

def _parse_relevant_indices(raw_response: str) -> set[int]:
    """Tolerant parsing of index numbers from the LLM response, allowing for
    extra text/markdown wrapping."""
    if "NONE" in raw_response.upper() and not re.search(r"\d", raw_response):
        return set()
    return {int(n) for n in re.findall(r"\d+", raw_response)}

def select_claims(job_description: str) -> list[str]:
    """Main entry point: a single batch LLM call, returning the final list of
    claim_ids (UUID strings) to include in this resume."""
    with get_session() as session:
        candidate_ids, governances = _resolve_candidate_ids(session)
        if not candidate_ids:
            return []
        
        contexts = fetch_experience_contexts(session, candidate_ids)
        tags_map = fetch_tags_for_claims(session, candidate_ids)

        user_prompt, index_to_claim_id = _build_batch_prompt(
            candidate_ids, contexts, governances, tags_map, job_description
        )
        if not index_to_claim_id:
            return []

        response = get_llm_client().generate_text(
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1000,
        )
        # print(response)
    relevant_indices = _parse_relevant_indices(response.text)
    selected_ids = [
        index_to_claim_id[i] for i in relevant_indices if i in index_to_claim_id
    ]
    return selected_ids
