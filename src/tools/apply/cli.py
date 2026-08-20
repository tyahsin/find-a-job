"""
apply/cli.py
Version: v4 (2026-08-20)

CLI entry point for the apply/ module -- wraps application_logger.py's
four functions (find_or_create_job/record_application/update_stage/
list_applications) as Typer commands. Uses a plain empty callback()
(Typer's standard group pattern, not invoke_without_command).

Command naming (user-confirmed 2026-08-20):
    apply create -> find_or_create_job(), with optional LLM-assisted
                    JD extraction path via --jd-file
    apply record -> record_application()
    apply update -> update_stage()
    apply list   -> list_applications()

(v2 change, 2026-08-20): the standalone `apply extract` command was
REMOVED -- absorbed into `apply create` via an LLM extraction +
one-shot confirmation flow, converging on a single find_or_create_job()
call regardless of path.

(v3 fix, 2026-08-20): added --jd-file (in addition to --jd-text at the
time) to read JD text from a file path, since --jd-text only accepts
literal pasted text -- passing a file path string to --jd-text does
NOT read the file (the shell does not expand it), it just sends the
path string itself to the LLM as if it were the JD body.

(v4 change, 2026-08-20, user-confirmed):
- --jd-text REMOVED entirely. In practice the user always has the JD
  saved to a file, never pastes raw JD text on the command line -- so
  --jd-text added a code path that was never actually going to be
  used, plus it was the direct cause of the v3 bug report (a file path
  string got treated as literal JD text). --jd-file is now the only
  way to trigger the LLM-assisted extraction path.
- When --jd-file is NOT provided, --company / --title / --location /
  --description are now all REQUIRED (previously only --company and
  --title were required). Rationale: without an LLM extraction pass,
  there is no other source for job_location/job_description, and
  leaving them silently optional meant a manually-created job row
  could end up missing fields that gap_analyzer.py (Step 3) will later
  depend on for JD-vs-tag comparison. source_url/date_posted/
  contact_name/contact_email remain optional in both paths since they
  are genuinely not always available (e.g. a JD posted without a named
  contact).
- All Typer-facing text (help strings, echo/prompt messages, error
  messages) converted from Chinese to English per user instruction.
  Docstrings remain English (they always were).
- The one-shot confirmation logic (show draft, accept-all / override
  fields / abort) is no longer implemented inline here -- it now calls
  core/confirmation.py's confirm_fields_once(), a generic helper
  extracted from this same pattern. See core/confirmation.py's
  docstring for why experience_rewriter.py's per-bullet regenerate loop
  was deliberately NOT folded into the same helper (different risk
  tier, different interaction shape).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer

from apply.application_logger import (
    find_or_create_job,
    list_applications,
    record_application,
    update_stage,
)
from apply.job_extractor import JOB_FIELDS, extract_job_fields
from core.confirmation import confirm_fields_once

app = typer.Typer(help="Job collection, application tracking, and notifications.")


@app.callback()
def callback() -> None:
    """apply/ module: job collection, application logging, and interview stage tracking."""


def _read_jd_file(jd_file: str) -> str:
    """Reads JD text from a file path (UTF-8). Exits cleanly with an
    error message if the file does not exist, rather than letting a
    raw FileNotFoundError traceback surface to the user.
    """
    file_path = Path(jd_file).expanduser()
    if not file_path.is_file():
        typer.echo(f"Error: file not found: {file_path}", err=True)
        raise typer.Exit(code=1)
    return file_path.read_text(encoding="utf-8")


@app.command("create")
def create_job(
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
) -> None:
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
        jd_text = _read_jd_file(jd_file)
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
        return

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


@app.command("record")
def record(
    job_id: str = typer.Option(..., "--job-id", help="The job_id to record an application for (see apply create)"),
    file_path: Optional[str] = typer.Option(None, "--file-path", help="Path to the final submitted CV yaml/PDF"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Notes"),
) -> None:
    """Records an actual application (record_application()). current_stage defaults to Applied."""
    application_id = record_application(job_id=job_id, file_path=file_path, notes=notes)
    typer.echo(f"application_id: {application_id}")


@app.command("update")
def update(
    application_id: str = typer.Option(..., "--application-id", help="The application_id to update"),
    new_stage: str = typer.Option(..., "--stage", help="New stage (see APPLICATION_CURRENT_STAGE_OPTIONS)"),
) -> None:
    """Updates application.current_stage (update_stage()). Transitioning to Interviewing auto-increments interview_round."""
    try:
        update_stage(application_id=application_id, new_stage=new_stage)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"application {application_id} -> {new_stage}")


@app.command("list")
def list_apps(
    stage: Optional[str] = typer.Option(None, "--stage", help="Filter to applications in this stage only"),
) -> None:
    """Lists application records (list_applications()), optionally filtered by --stage."""
    rows = list_applications(current_stage=stage)
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
        typer.echo(line)


if __name__ == "__main__":
    app()
