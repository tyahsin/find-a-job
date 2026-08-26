"""
track/cli.py
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from sqlalchemy import cast, String
from pathlib import Path
from typing import Optional

import typer

from core.source_reader import read_source
from core.db import JobORM, ApplicationORM, APPLICATION_CURRENT_STAGE_OPTIONS, get_session, fuzzy_find_candidates, create_row, select_or_create
from core.llm_reviewer import confirm_fields_multiple
from core.source_reader import FileReadError, UnsupportedFileTypeError, read_source

from onboard.cli import (
    _select_from_fuzzy_candidates,
    _select_from_menu,
    _create_table,
)
from track.prompts import JOB_FIELDS, extract_job_fields


app = typer.Typer(help="Collect jobs and track applications.")


def _select_existing_job(session, column: str, keyword: str, require_choice: bool = True) -> JobORM:
    label_field = ["job_id", "company_name", "job_title"]

    candidates = fuzzy_find_candidates(session, JobORM, column, keyword)
    if candidates:
        chosen = _select_from_fuzzy_candidates(candidates, label_field, require_choice)
        if chosen is not None:
            return chosen


def _create_job_from_user(
    session,
    company_name: str,
    job_title: str,
    job_location: str,
    job_description: str,
    source_url: str,
    date_posted: date,
    contact_name: str,
    contact_email: str,
) -> JobORM:
    return create_row(
        session,
        JobORM,
        {
            "company_name": company_name,
            "job_title": job_title,
            "job_location": job_location,
            "job_description": job_description,
            "source_url": source_url,
            "date_posted": date_posted,
            "contact_name": contact_name,
            "contact_email": contact_email,
        }
    )


def _create_job_from_source(session, source) -> JobORM:
    jd_text = read_source(source)
    result = extract_job_fields(jd_text, source)
    edited = confirm_fields_multiple(list(result.values()), list(result.keys()), [""] * len(result))
    finalized = {item[0]: item[1] for item in edited}
    return _create_job_from_user(session, *finalized)


def _resolve_job(session, source: str | None = None) -> JobORM:
    if source:
        while True:
            try:
                job = _create_job_from_source(session, source)
                break
            except (FileReadError, UnsupportedFileTypeError) as exc:
                source = typer.prompt("Enter file path (.txt / .md / .pdf) or URL")
        return job
    else:
        job_fields = {}
        for field in JOB_FIELDS:
            prompt_text = f"{field.replace('_', ' ').capitalize()}"
            if  "date" in field:
                prompt_text += f" (in format yyyy-MM-dd; optional)" 
            job_fields[field] = typer.prompt(f"{prompt_text}", default="", show_default=False)
        return _create_job_from_user(session, *job_fields)


@app.command("add-job")        
def add_job(
    column: str = typer.Option(..., "-c", "--column"), 
    keyword: str = typer.Option(..., "-k", "--keyword"), 
):
    with get_session() as session:
        job = _select_existing_job(session, column, keyword)
        if not job:
            source = typer.prompt("Enter file path (.txt / .md / .pdf) or URL", default="", show_default=False)
            job = _resolve_job(session, source)
    return job


@app.command("list-jobs")
def list_jobs(
    column: str = typer.Option(..., "-c", "--column"), 
    keyword: str = typer.Option(..., "-k", "--keyword"), 
):
    with get_session() as session:
        _select_existing_job(session, column, keyword, require_choice=False)


def _select_existing_application(session, column: str, keyword: str):
    job = add_job(column, keyword)
    application = (
        session.query(ApplicationORM).
        join(JobORM, ApplicationORM.job_id==JobORM.job_id).
        filter(JobORM.job_id==job.job_id).
        one_or_none()
    )
    return application, job


def _create_application(session, job):
    file_folder = None
    while True:
        file_folder = typer.prompt("File folder/location?")
        if Path(file_folder).exists():
            break
        else:
            print("Wrong input. Please try again.")
    notes = typer.prompt("Any notes?", default="", show_default=False)
    return create_row(
        session,
        ApplicationORM,
        {
            "job_id": job.job_id,
            "file_folder": file_folder,
            "current_stage": "Applied",
            "interview_round": 0,
            "notes": notes,
        }
    )


def _update_application(application):

    application.current_stage = _select_from_menu(APPLICATION_CURRENT_STAGE_OPTIONS, "Which stage has this application progressed to?")
    if application.current_stage == "Interviewing":
        application.interview_round = (application.interview_round or 0) + 1
    application.notes = typer.prompt("Any notes?", default="")
    application.updated_at = datetime.now(timezone.utc)

    return application  


@app.command("update-application")
def _resolve_application(    
    column: str = typer.Option(..., "-c", "--column"), 
    keyword: str = typer.Option(..., "-k", "--keyword"), 
):
    with get_session() as session:
        application, job = _select_existing_application(session, column, keyword)
        if not application:
            application = _create_application(session, job)
        else:
            application = _update_application(application)


@app.command("list-applications") # by other pillars
def list_applications():
    with get_session() as session:
        stage = _select_from_menu(APPLICATION_CURRENT_STAGE_OPTIONS, "Select a stage to lookup")
        rows = (
                session.query(
                JobORM.company_name, 
                JobORM.job_title, 
                cast(ApplicationORM.updated_at, String).label("updated_at"),
                ApplicationORM.current_stage, 
                cast(ApplicationORM.interview_round, String).label("interview_round"), 
                ApplicationORM.notes, 
            ).
            join(JobORM, ApplicationORM.job_id==JobORM.job_id).
            filter(ApplicationORM.current_stage==stage).
            all()
        )
        if not rows:
            print("No matching application records.")
            return

        title = f"Found {len(rows)} existing records"
        columns = ["company_name", "job_title", "updated_at", "current_stage", "interview_round", "notes"]
        _create_table([tuple(row) for row in rows], columns, title)


if __name__ == "__main__":
    app()
