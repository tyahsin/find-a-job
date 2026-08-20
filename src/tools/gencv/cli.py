"""
gencv/cli.py
Version: v3 (2026-08-19)

Changelog:
- v3 (2026-08-19): Added two new REQUIRED options, --company-name and
  --job-title, threaded straight through to pipeline.generate_cv()'s
  new required company_name/job_title parameters (see pipeline.py v11
  changelog) -- these are needed to compute the per-application PDF
  output path (resume_builder.build_cv_pdf_path(): folder
  {OUTPUT_DIR}/{job_title}_{company_name}/, filename
  cv_{name}_{job_title}_{company_name}.pdf). There is no fallback/
  extraction path for these two (unlike --headline, which falls back to
  JD extraction) -- the user must supply them explicitly on every run.
- v2 (2026-08-15):
  1. --headline changed from required to optional (default None). If
     omitted, pipeline.generate_cv() will attempt to extract it from the
     job description text via identity_selector.extract_headline_from_jd().
  2. --name changed from a hardcoded default ("Ya Hsin Tsai") to optional
     with no default -- name now comes from the selected identity.
  3. Full English-only pass -- every CLI-facing string converted from
     Chinese to English, per the project-wide LANGUAGE POLICY.
- v1 (2026-08-12, inferred): Initial outer CLI entrypoint, --out removed.

Outer CLI entrypoint for the gencv package. Operating style matches
onboard/cli.py -- day-to-day usage runs this file directly, no need to
touch pipeline.py / experience_matcher.py or other internal modules.

Usage (run from src/tools/):
python -m gencv.cli generate --job-description-file jd.txt --company-name "Acme Corp" --job-title "Software Engineer"
python -m gencv.cli generate --job-description-file jd.txt --company-name "Acme Corp" --job-title "Data Scientist" --headline "Data Scientist"
python -m gencv.cli mark-reviewed
"""

import pathlib

import typer

from gencv.pipeline import generate_cv as gen_cv

app = typer.Typer(help="""gencv: JD -> tailored CV PDF pipeline.""")


@app.command("generate-cv")
def generate_cv(
    job_id: str = typer.Option(
        ..., "--job-id", help="Job ID (required -- used to build the PDF output path)"
    ),
    name: str = typer.Option(
        None,
        "--name",
        help="Display name override for this run (optional, defaults to the selected identity's name)",
    ),
    headline: str = typer.Option(
        None,
        "--headline",
        help="Resume headline, e.g. target job title (optional -- extracted from the JD text if omitted)",
    ),
) -> str:
    """Select relevant experiences -> confirm rewrites -> render to PDF -> log the application."""

    pdf_path = gen_cv(
        job_id=job_id,
        headline=headline,
        name=name,
    )
    typer.secho(f"CV generated: {str(pdf_path)}", fg=typer.colors.GREEN)
    return pdf_path.parent


if __name__ == "__main__":
    app()
