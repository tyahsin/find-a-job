"""
gencv/cli.py
"""

import typer

from track.cli import add_job
from gencv.resume_builder import generate_cv

app = typer.Typer(help="""gencv: JD -> tailored CV PDF pipeline.""")

@app.command("generate-cv")
def generate_cv_cli(
    column: str = typer.Option("job_title", "-c", "--column", help="column to search"),
    keyword: str = typer.Option("data", "-k", "--keyword", help="keyword for column to search; support fuzzy search"),
    name: str = typer.Option(None, "-n", "--name", help="Display name override for this run (optional)"),
    headline: str = typer.Option(None, "-h", "--headline", help="Display headline override for this run (optional)")
):
    job = add_job(column, keyword)
    pdf_path = generate_cv(job, headline, name)
    typer.secho(f"CV generated: {str(pdf_path)}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
