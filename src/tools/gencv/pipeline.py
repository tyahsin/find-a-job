"""
pipeline.py
Version: v7 (2026-08-16)

Changelog:
- v7 (2026-08-16): Simplified to match resume_builder.py v12's
  build_education_entries() signature change (dropped
  degree_ids_with_projects param). This pipeline no longer needs to call
  build_project_entries() early just to obtain that set -- it now calls
  build_education_entries(session, relevant_degrees) directly, with no
  ordering dependency on project assembly. This also fixes the QA-reported
  issue where education_selector.py's orphan-degree path (Path B) was
  effectively dead code: previously, build_education_entries() filtered
  relevant_degrees down to only degrees with a selected project underneath
  them, which discarded every orphan degree select_relevant_degrees()
  worked to surface. That filter is gone in resume_builder.py v12, so this
  pipeline's existing call to select_relevant_degrees() (step 5 below) now
  actually has an effect on the final resume, as originally intended.
- v6 (2026-08-16): [SUPERSEDED by v7] Added build_project_entries() call
  solely to compute degree_ids_with_projects for build_education_entries().
- v5 (2026-08-15): Added summary_writer.draft_and_confirm_summary().
- v4 (2026-08-15): education_selector.select_relevant_degrees() added.
- v3 (2026-08-15): headline optional (extracted from JD when not provided).
- v2 (2026-08-15): identity/network selection added before select_claims().
- v1 (2026-08-10, inferred): Initial JD -> PDF pipeline.

內部主流程（不對外暴露，對外唯一入口是 gencv/cli.py）：輸入 job description，輸出最終 PDF 路徑。

流程：
1. identity_selector.select_identity_from_jd() -- 選出這次要用的身份
2. identity_selector.select_networks_from_jd() -- 選出符合工種的連結
3. identity_selector.extract_headline_from_jd() -- 若 headline 未提供，從 JD 擷取職稱
4. experience_matcher.select_claims() -- 篩選相關經歷
5. education_selector.select_relevant_degrees() -- 決定 education section 顯示哪些學位
   （Path A: claim-backed 聯集 Path B: orphan degree LLM 判斷，無需再被 project 命中篩選）
6. experience_rewriter.rewrite_and_confirm() -- 融合 STAR + 貼近 JD 用語，逐條人工確認
7. resume_builder.build_education_entries() -- 組出 education entry 清單（summary_writer 需要這份資料）
8. summary_writer.draft_and_confirm_summary() -- 總結整份履歷 + 視情況 bridge JD gap，單次確認
9. resume_builder.build_resume_yaml() -- 組裝履歷內容（含 summary，內部重新跑一次 project/education/experience 組裝）
10. resume_builder.render_pdf() -- 呼叫 rendercv CLI 排版編譯成 PDF
11. application_log.log_generated_output() -- 記錄這次投遞
"""

import pathlib
import tempfile

from gencv.identity_selector import (
    select_identity_from_jd,
    select_networks_from_jd,
    extract_headline_from_jd,
)
from gencv.education_selector import select_relevant_degrees
from gencv.experience_matcher import select_claims
from gencv.experience_rewriter import rewrite_and_confirm
from gencv.summary_writer import draft_and_confirm_summary
from gencv.resume_builder import build_resume_yaml, build_education_entries, render_pdf
from gencv.application_log import log_generated_output
from core.db import get_session


def generate_cv(
    job_description: str,
    headline: str | None = None,
    name: str | None = None,
) -> tuple[pathlib.Path, str]:
    """JD -> PDF 單一入口。

    Returns:
        (PDF 路徑, 該次投遞紀錄的 output_id)
    """
    identity = select_identity_from_jd(job_description)
    networks = select_networks_from_jd(job_description)

    if headline is None:
        headline = extract_headline_from_jd(job_description)

    selected_claim_ids = select_claims(job_description)

    if not selected_claim_ids:
        raise ValueError("篩選後沒有任何經歷通過（可能被治理層擋下，或 LLM 判定皆不相關）。")

    with get_session() as session:
        relevant_degrees = select_relevant_degrees(session, selected_claim_ids, job_description)

    final_descriptions = rewrite_and_confirm(selected_claim_ids, job_description)

    with get_session() as session:
        education_entries_for_summary = build_education_entries(session, relevant_degrees)

    summary = draft_and_confirm_summary(
        job_description,
        list(final_descriptions.values()),
        education_entries_for_summary,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        resume_yaml_path = pathlib.Path(tmp_dir) / "resume_content.yaml"
        build_resume_yaml(
            selected_claim_ids,
            identity,
            str(resume_yaml_path),
            headline=headline,
            networks=networks,
            relevant_degrees=relevant_degrees,
            description_overrides=final_descriptions,
            name=name,
            summary=summary,
        )

        pdf_out_path = render_pdf(cv_yaml_path=resume_yaml_path)

    output_id = log_generated_output(
        job_description=job_description,
        used_claim_ids=selected_claim_ids,
        output_type="pdf",
    )

    print(f"[OK] 已記錄這次投遞，output_id={output_id}（尚待你正式複核）")

    return pdf_out_path, output_id
