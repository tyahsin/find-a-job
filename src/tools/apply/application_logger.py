"""
apply/application_logger.py
Version: v1 (2026-08-19)

Replaces gencv/application_logger.py (which wrote to the now-removed
generated_outputs table via log_generated_output()/mark_reviewed()).

Design principle (per user decision, 2026-08-19): gencv/pipeline.py's
generate_cv() is a pure JD -> PDF draft producer with NO tracking side
effect -- it can be re-run any number of times (including after the user
hand-edits the yaml and re-runs rendercv independently) without polluting
any record. Recording an actual application only happens here, via a
deliberate, explicit call, once the user has decided a given CV version
is the one they are actually submitting.

Corresponds to core/db.py's JobORM (job table) and ApplicationORM
(application table) -- see 001_init_schema.sql's apply/ module addition.

interview_round auto-increment: business logic stays here, in Python, not
in a database trigger -- consistent with this project's existing
fill_pending_support_levels() philosophy in core/db.py. update_stage()
increments application.interview_round by 1 every time current_stage
transitions TO "Interviewing" (including re-entering it, e.g. a second
round after a gap) -- it does not increment on every call, only on that
specific transition.
"""

from __future__ import annotations

import uuid

from core.db import (
    APPLICATION_CURRENT_STAGE_OPTIONS,
    ApplicationORM,
    JobORM,
    get_session,
    select_or_create,
)


def find_or_create_job(
    company_name: str,
    job_title: str,
    job_location: str | None = None,
    job_description: str | None = None,
    source_url: str | None = None,
    date_posted=None,
    contact_name: str | None = None,
    contact_email: str | None = None,
) -> str:
    """Finds an existing job row by exact (company_name, job_title) match,
    or creates a new one. Returns the job_id (str).

    Uses core.db.select_or_create() -- exact-match semantics, same as the
    rest of this codebase's onboard flows (see onboard/cli.py's
    _resolve_company()/_resolve_school() for the equivalent pattern with
    fuzzy matching layered on top at the CLI level, if ever needed here).
    """
    with get_session() as session:
        job = select_or_create(
            session,
            JobORM,
            match_filters={"company_name": company_name, "job_title": job_title},
            create_values={
                "company_name": company_name,
                "job_title": job_title,
                "job_location": job_location,
                "job_description": job_description,
                "source_url": source_url,
                "date_posted": date_posted,
                "contact_name": contact_name,
                "contact_email": contact_email,
            },
        )
        job_id = str(job.job_id)
    return job_id


def record_application(
    job_id: str,
    file_path: str | None = None,
    notes: str | None = None,
) -> str:
    """Records an actual application -- called ONLY when the user has
    decided to submit a given CV version (possibly hand-edited after
    generate_cv() produced a first draft). current_stage defaults to
    "Applied". Returns the new application_id (str).

    Deliberately NOT called automatically by gencv/pipeline.py -- see
    module docstring.
    """
    with get_session() as session:
        application = ApplicationORM(
            job_id=uuid.UUID(job_id) if isinstance(job_id, str) else job_id,
            file_path=file_path,
            current_stage="Applied",
            interview_round=0,
            notes=notes,
        )
        session.add(application)
        session.flush()
        application_id = str(application.application_id)
    return application_id


def update_stage(application_id: str, new_stage: str) -> None:
    """Updates application.current_stage. Validates new_stage against the
    same fixed vocabulary enforced by the database CHECK constraint
    (core.db.APPLICATION_CURRENT_STAGE_OPTIONS) -- fails fast here with a
    clear Python-level error rather than letting an invalid value reach
    the database and raise an opaque IntegrityError.

    interview_round auto-increment: if new_stage == "Interviewing",
    increments application.interview_round by 1, regardless of what the
    previous stage was -- this is CLI-layer business logic (not a
    database trigger), consistent with this project's existing
    "business logic stays in Python" principle (see core/db.py's
    fill_pending_support_levels()).
    """
    if new_stage not in APPLICATION_CURRENT_STAGE_OPTIONS:
        raise ValueError(
            f"Invalid current_stage: {new_stage!r}. Must be one of: "
            f"{', '.join(APPLICATION_CURRENT_STAGE_OPTIONS)}"
        )

    app_uuid = uuid.UUID(application_id) if isinstance(application_id, str) else application_id

    with get_session() as session:
        application = session.get(ApplicationORM, app_uuid)
        if application is None:
            raise ValueError(f"No application found for application_id: {application_id}")

        application.current_stage = new_stage
        if new_stage == "Interviewing":
            application.interview_round = (application.interview_round or 0) + 1


def list_applications(current_stage: str | None = None) -> list[dict]:
    """Query helper -- returns a flat list of application summaries,
    optionally filtered by current_stage. Each entry includes the joined
    job info (company_name/job_title) so callers don't need a separate
    lookup. Intended for CLI display (e.g. a future `apply list` command)
    -- returns plain dicts, not ORM objects, so results remain usable
    after the session closes.
    """
    with get_session() as session:
        query = session.query(ApplicationORM, JobORM).join(JobORM, ApplicationORM.job_id == JobORM.job_id)
        if current_stage is not None:
            query = query.filter(ApplicationORM.current_stage == current_stage)

        rows = query.all()
        results = []
        for application, job in rows:
            results.append(
                {
                    "application_id": str(application.application_id),
                    "company_name": job.company_name,
                    "job_title": job.job_title,
                    "current_stage": application.current_stage,
                    "interview_round": application.interview_round,
                    "file_path": application.file_path,
                    "notes": application.notes,
                    "created_at": application.created_at,
                    "updated_at": application.updated_at,
                }
            )
    return results
