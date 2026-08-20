import typer
import click

from onboard.cli import app as onboard_app
from gencv.cli import app as gencv_app, gen_cv
from track.cli import app as track_app, add_job, add_application


app = typer.Typer(help="Apply for a job in the workflow")
app.add_typer(onboard_app, name="onboard")
app.add_typer(gencv_app, name="gencv")
app.add_typer(track_app, name="track")


@app.command("apply")
def apply(
    jd_file: str = typer.Option(
        None, "--jd-file", help="Path to a JD text file (triggers LLM extraction + one-shot confirmation)"
    ),
) -> None:
    job_id = add_job(jd_file=jd_file)
    file_folder = gen_cv(job_id)
    add_application(job_id, file_folder)

    return


if __name__ == "__main__":
    app()