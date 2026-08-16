"""
cli.py
Version: v2 (2026-08-15)

Changelog:
- v2 (2026-08-15):
  1. --headline changed from required to optional (default None). If
     omitted, pipeline.generate_cv() will attempt to extract it from the
     job description text via identity_selector.extract_headline_from_jd()
     -- see pipeline.py v3 changelog. If extraction also finds nothing,
     headline stays unset and is omitted from the final YAML.
  2. --name changed from a hardcoded default ("Ya Hsin Tsai") to optional
     with no default. Rationale: name now comes from the selected identity
     (see resume_builder.py v2 / pipeline.py -- identity.name is the source
     of truth), --name is purely an optional per-run display override,
     matching build_resume_yaml()'s v2 signature intent. A hardcoded
     fallback name no longer makes sense once multiple identities exist.
  3. Full English-only pass -- every CLI-facing string (help text, error
     messages, success messages) converted from Chinese to English, per
     the project-wide LANGUAGE POLICY established in onboard/prompts.py
     (the entire interface is English-only, no exceptions).
- v1 (2026-08-12, inferred): Initial outer CLI entrypoint, --out removed.

Outer CLI entrypoint for the gencv package. Operating style matches
onboard/cli.py -- day-to-day usage runs this file directly, no need to
touch pipeline.py / experience_matcher.py or other internal modules.

Usage (run from src/tools/):
python -m gencv.cli generate --job-description-file jd.txt
python -m gencv.cli generate --job-description-file jd.txt --headline "Data Scientist"
python -m gencv.cli mark-reviewed <output_id>
"""

import pathlib

import typer

from gencv.pipeline import generate_cv
from gencv.application_log import mark_reviewed as _mark_reviewed

app = typer.Typer(help="gencv: Job Description -> Tailored CV PDF")


@app.command("generate")
def generate(
    job_description_file: str = typer.Option(
        None, "--job-description-file", help="Path to a JD text file"
    ),
    job_description: str = typer.Option(
        None, "--job-description", help="Paste JD text directly"
    ),
    headline: str = typer.Option(
        None,
        "--headline",
        help="Resume headline, e.g. target job title (optional -- extracted from the JD text if omitted)",
    ),
    name: str = typer.Option(
        None,
        "--name",
        help="Display name override for this run (optional, defaults to the selected identity's name)",
    ),
):
    """Select relevant experiences -> confirm rewrites -> render to PDF -> log the application."""
    jd_text = job_description
    if job_description_file:
        jd_text = pathlib.Path(job_description_file).read_text(encoding="utf-8")

    if not jd_text:
        typer.secho("Must provide either --job-description or --job-description-file", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    result_path, output_id = generate_cv(
        job_description=jd_text,
        headline=headline,
        name=name,
    )

    typer.secho(f"CV generated: {result_path}", fg=typer.colors.GREEN)


@app.command("mark-reviewed")
def mark_reviewed(output_id: str = typer.Argument(..., help="The output_id (UUID) to mark as reviewed")):
    """Mark a generated output's human_review_status as True."""
    _mark_reviewed(output_id)
    typer.secho(f"Marked {output_id} as reviewed", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
