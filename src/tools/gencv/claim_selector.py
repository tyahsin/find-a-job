
import re
from sqlalchemy import text
from core.llm_client import get_llm_client

SQL_QUERY = """
SELECT
    degree.degree_id IS NULL AS is_work_track,
    COALESCE(company.company_name, school.school_name) AS org_name,
    COALESCE(position.department, degree.degree_level) AS org_sub_name,
    COALESCE(position.position_id, degree.degree_id) AS display_id,
    COALESCE(position.title, degree.degree_name) AS display_name,
    COALESCE(position.start_date, degree.start_date) AS start_date,
    COALESCE(position.end_date, degree.end_date) AS end_date,
    COALESCE(position.responsibility, degree.degree_description) AS display_desc,
    project.project_id,
    project.project_name,
    project.project_context,
    problem.problem_description,
    problem.stakeholder_impact,
    action.action_description,
    action.role_scope,
    claim.claim_id,
    claim.claim_description,
    governance.use_policy,
    governance.notes_for_llm
FROM claim
JOIN governance ON claim.claim_id = governance.claim_id
JOIN action ON claim.action_id = action.action_id
JOIN problem ON action.problem_id = problem.problem_id
JOIN project ON problem.project_id = project.project_id
LEFT JOIN position ON project.position_id = position.position_id
LEFT JOIN company ON position.company_id = company.company_id
LEFT JOIN degree ON project.degree_id = degree.degree_id
LEFT JOIN school ON degree.school_id = school.school_id
WHERE governance.allowed_for_generation = 't'
"""

def _build_claim_selection_prompt(idx: int, ctx: dict):
    return (
        f"Expereince [{idx}]\n"
        f"    Background: {ctx["project_context"]}\n"
        f"    Problem: {ctx['problem_description']}\n"
        f"    Action: {ctx['action_description']}\n"
        f"    Result: {ctx['claim_description']}\n"
        f"    Usage policy: {ctx["use_policy"]}"
    )

RELEVANCE_SYSTEM_PROMPT = """You are a resume screening assistant. You will be given a job description
and a list of the candidate's past experience records (each with an index,
background, project context, problem, action, result, and that record's usage policy).


Judge whether each experience record is relevant to this job posting and
worth including in a resume tailored to it.


A record is relevant if EITHER of the following is true:
- The business domain or work setting matches the job description (e.g.
  same industry, same type of stakeholder collaboration), OR
- The technical skills, methods, or tools demonstrated in the record match
  skills explicitly required or mentioned in the job description -- even if
  the project's domain or setting (e.g. academic project vs. corporate
  role) differs from the job posting's context.


Do not require both domain match AND skill match simultaneously. A
technically strong record demonstrating a required skill (e.g. a specific
ML method, a specific tool, hands-on model development) should be marked
relevant even if it comes from an academic or personal project, as long as
the skill itself is explicitly called for in the job description.


If a record's usage policy (use_policy) explicitly excludes this type of
job, always treat it as not relevant, regardless of content match.

Output format: output only the index numbers of relevant records, one per
line, no other text, no explanation beyond the numbers. If none are
relevant, output NONE."""

def _parse_relevant_indices(raw_response: str) -> set[int]:
    """Tolerant parsing of index numbers from the LLM response, allowing for
    extra text/markdown wrapping."""
    if "NONE" in raw_response.upper() and not re.search(r"\d", raw_response):
        return set()
    return {int(n) for n in re.findall(r"\d+", raw_response)}

def select_claims(session, job_description: str) -> list[str]:
    
    # contexts: all allowed claims with attributes
    rows = (
        session.execute(
            text(SQL_QUERY),
        ).mappings().all()
    )
    if not rows:
        return None, None

    contexts = [dict(row) for row in rows]

    # llm select context into ranking
    lines = []
    for idx, ctx in enumerate(contexts):
        lines.append(_build_claim_selection_prompt(idx, ctx))
    experiences_block = "\n\n".join(lines)

    user_prompt = f"""# Job Description
{job_description}

# Past experience records ({len(contexts)} total)
{experiences_block}"""

    # Send to prompt for selection
    print(f"  Selecting experiences...\n")
    response = get_llm_client().generate_text(
        system_prompt=RELEVANCE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=1000,
    )

    # Parse response
    relevant_indices = _parse_relevant_indices(response.text)

    # Return selected claims and not selected role
    selected = {
        "work_track": [],
        "academic_track": []
    }
    for idx, ctx in enumerate(contexts):
        if idx in relevant_indices:
            if ctx["is_work_track"] is True:
                selected["work_track"].append(ctx)
            else:
                selected["academic_track"].append(ctx)
    return selected


