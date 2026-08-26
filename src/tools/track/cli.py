"""
track/cli.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer

from track.job_app_tracker import (
    find_or_create_job,
    list_applications as list_apps,
    record_application,
    update_stage,
)
from track.job_extractor import JOB_FIELDS, extract_job_fields
from core.source_reader import read_source
from core.llm_reviewer import confirm_fields_once

app = typer.Typer(help="Collect jobs and track applications.")


@app.command("find-job")
def find_job(
    company_name: str = typer.Option(..., "--company-name", help="Company name"),
    job_title: str = typer.Option(..., "--job-title", help="Job title"),
) -> str: 
    """
    Finds an existing job row by exact (company_name, job_title) match. Returns the job_id (str).
    """
    job_id = find_or_create_job(
        company_name=company_name,
        job_title=job_title,
    )

    if job_id is None:
        typer.echo(f"job_id not found for your entry")
    else:
        typer.echo(f"job_id: {job_id}")


@app.command("add-job")
def add_job(
    company_name: Optional[str] = typer.Option(None, "--company", help="Company name"),
    job_title: Optional[str] = typer.Option(None, "--title", help="Job title"),
    job_location: Optional[str] = typer.Option(None, "--location", help="Job location"),
    job_description: Optional[str] = typer.Option(None, "--description", help="Full JD text"),
    source_url: Optional[str] = typer.Option(None, "--url", help="Job posting URL"),
    date_posted: Optional[str] = typer.Option(None, "--date-posted", help="Date posted, YYYY-MM-DD"),
    contact_name: Optional[str] = typer.Option(None, "--contact-name", help="Contact person's name"),
    contact_email: Optional[str] = typer.Option(None, "--contact-email", help="Contact person's email"),
    jd_file: Optional[str] = typer.Option(
        None, "--jd-file", help="Path to a JD text file (triggers LLM extraction + one-shot confirmation)"
    ),
) -> str:
    """Finds or creates a job row (find_or_create_job()). Matches
    exactly on (company_name, job_title).

    Two modes:
    - --jd-file provided: read file -> LLM extraction -> one-shot
      confirm/override -> write.
    - --jd-file not provided: write directly from CLI args. In this
      mode --company/--title/--location/--description are all required
      (there is no other source for them without LLM extraction).
    """
    if jd_file:
        jd_text = read_source(jd_file)
        extracted = extract_job_fields(jd_text=jd_text, source_url=source_url)
        confirmed = confirm_fields_once(
            fields=extracted,
            field_order=JOB_FIELDS,
            header="LLM-extracted job fields (fields not found in the JD are None):",
        )
        if confirmed is None:
            typer.echo("Aborted. Nothing was written to the database.")
            raise typer.Exit(code=0)

        job_id = find_or_create_job(
            company_name=confirmed["company_name"],
            job_title=confirmed["job_title"],
            job_location=confirmed["job_location"],
            job_description=confirmed["job_description"],
            source_url=confirmed["source_url"],
            date_posted=confirmed["date_posted"],
            contact_name=confirmed["contact_name"],
            contact_email=confirmed["contact_email"],
        )
        typer.echo(f"job_id: {job_id}")
        return job_id

    missing = [
        name
        for name, value in (
            ("--company", company_name),
            ("--title", job_title),
            ("--location", job_location),
            ("--description", job_description),
        )
        if not value
    ]
    if missing:
        typer.echo(
            "Error: without --jd-file, the following options are required: "
            + ", ".join(missing),
            err=True,
        )
        raise typer.Exit(code=1)

    parsed_date = date.fromisoformat(date_posted) if date_posted else None
    job_id = find_or_create_job(
        company_name=company_name,
        job_title=job_title,
        job_location=job_location,
        job_description=job_description,
        source_url=source_url,
        date_posted=parsed_date,
        contact_name=contact_name,
        contact_email=contact_email,
    )
    typer.echo(f"job_id: {job_id}")
    return job_id


@app.command("add-application")
def add_application(
    job_id: str = typer.Option(..., "--job-id", help="The job_id to record an application for"),
    file_folder: Optional[str] = typer.Option(None, "--file-folder", help="Path to the final submitted CV yaml/PDF folder"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Notes"),
) -> None:
    """Records an actual application (record_application()). current_stage defaults to Applied."""
    application_id = record_application(job_id=job_id, file_folder=file_folder, notes=notes)
    typer.echo(f"application_id: {application_id}")


@app.command("update-application")
def update_application(
    application_id: str = typer.Option(..., "--application-id", help="The application_id to update"),
    new_stage: str = typer.Option(..., "--stage", help="New stage (see APPLICATION_CURRENT_STAGE_OPTIONS)"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Notes"),
) -> None:
    """Updates application.current_stage (update_stage()). Transitioning to Interviewing auto-increments interview_round."""
    try:
        update_stage(application_id=application_id, new_stage=new_stage, notes=notes)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"application {application_id} -> {new_stage}")


@app.command("list-applications")
def list_applications(
    stage: Optional[str] = typer.Option(None, "--stage", help="Filter to applications in this stage only"),
) -> None:
    """Lists application records (list_applications()), optionally filtered by --stage."""
    rows = list_apps(current_stage=stage)
    if not rows:
        typer.echo("No matching application records.")
        return
    for row in rows:
        stage_val = row["current_stage"]
        company_val = row["company_name"]
        title_val = row["job_title"]
        round_val = row["interview_round"]
        app_id_val = row["application_id"]
        line = "[" + stage_val + "] " + company_val + " - " + title_val
        line += " (round " + str(round_val) + ", application_id=" + app_id_val + ")" 
        if stage_val in ["Offer"]:
            typer.secho(line, fg=typer.colors.GREEN)
        elif stage_val in ["Rejected", "Ghosted"]:
            typer.secho(line, fg=typer.colors.RED)
        else:
            typer.secho(line, fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
