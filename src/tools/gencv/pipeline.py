"""
pipeline.py
內部主流程(不對外暴露,對外唯一入口是 gencv/cli.py):輸入 job description,輸出最終 PDF 路徑。

流程:
  1. experience_matcher.select_claims() -- 篩選相關經歷(治理過濾 + 單次批次 LLM 判斷)
  2. experience_rewriter.rewrite_and_confirm() -- 逐條 LLM 輕度改寫 + 人工逐句確認
  3. resume_builder.build_resume_yaml() -- 組裝履歷內容
  4. resume_builder.render_pdf() -- 呼叫 rendercv CLI 排版編譯成 PDF
  5. application_log.log_generated_output() -- 記錄這次投遞

前置需求:pip install rendercv,且 templates/design.yaml / locale.yaml / settings.yaml
三份凍結設定檔已就位。
"""

import pathlib
import tempfile

from gencv.experience_matcher import select_claims
from gencv.experience_rewriter import rewrite_and_confirm
from gencv.resume_builder import build_resume_yaml, render_pdf
from gencv.application_log import log_generated_output


def generate_cv(
    job_description: str,
    headline: str,
    name: str = "Ya Hsin Tsai",
    pdf_out: str = "output/CV.pdf",
) -> tuple[pathlib.Path, str]:
    """JD -> PDF 單一入口。

    Returns:
        (PDF 路徑, 該次投遞紀錄的 output_id)
    """
    selected_claim_ids = select_claims(job_description)

    if not selected_claim_ids:
        raise ValueError("篩選後沒有任何經歷通過(可能被治理層擋下,或 LLM 判定皆不相關)。")

    final_descriptions = rewrite_and_confirm(selected_claim_ids, job_description)

    with tempfile.TemporaryDirectory() as tmp_dir:
        resume_yaml_path = pathlib.Path(tmp_dir) / "resume_content.yaml"
        build_resume_yaml(
            selected_claim_ids, headline, name, str(resume_yaml_path),
            description_overrides=final_descriptions,
        )

        pdf_out_path = pathlib.Path(pdf_out)
        render_pdf(cv_yaml_path=resume_yaml_path, pdf_out_path=pdf_out_path)

    output_id = log_generated_output(
        job_description=job_description,
        used_claim_ids=selected_claim_ids,
        output_type="pdf",
    )
    print(f"[OK] 已記錄這次投遞,output_id={output_id}(尚待你正式複核)")

    return pdf_out_path, output_id
