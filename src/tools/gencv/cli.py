"""
cli.py
gencv 的外層 CLI 入口,對外操作風格對齊 onboard/cli.py -- 日常使用直接跑這個檔案,
不需要碰 pipeline.py / experience_matcher.py 等內部模組。

用法(在 src/tools/ 目錄下執行):
    python -m gencv.cli generate --job-description-file jd.txt --headline "Data Scientist" --out output/CV.pdf
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
        None, "--job-description-file", help="JD 文字檔路徑"
    ),
    job_description: str = typer.Option(
        None, "--job-description", help="直接貼 JD 文字"
    ),
    headline: str = typer.Option(..., "--headline", help="履歷標題,例如職位名稱"),
    name: str = typer.Option("Ya Hsin Tsai", "--name", help="姓名"),
    out: str = typer.Option("output/CV.pdf", "--out", help="最終 PDF 輸出路徑"),
):
    """篩選相關經歷 -> 逐條改寫確認 -> 排版編譯成 PDF -> 記錄投遞紀錄。"""
    jd_text = job_description
    if job_description_file:
        jd_text = pathlib.Path(job_description_file).read_text(encoding="utf-8")

    if not jd_text:
        typer.secho("必須提供 --job-description 或 --job-description-file", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    result_path, output_id = generate_cv(
        job_description=jd_text,
        headline=headline,
        name=name,
        pdf_out=out,
    )
    typer.secho(f"CV generated: {result_path}", fg=typer.colors.GREEN)


@app.command("mark-reviewed")
def mark_reviewed(output_id: str = typer.Argument(..., help="要標記為已複核的 output_id(UUID)")):
    """把某次投遞紀錄的 human_review_status 標記為 True。"""
    _mark_reviewed(output_id)
    typer.secho(f"已將 {output_id} 標記為已複核", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
