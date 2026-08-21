"""
pipeline.py
Version: v11 (2026-08-19)

Changelog:
- v11 (2026-08-19): Per resume_builder.py v17 changes:
  1. generate_cv() gained two new required parameters, company_name and
     job_title -- needed to compute the PDF output path via the new
     resume_builder.build_cv_pdf_path() (folder:
     {OUTPUT_DIR}/{job_title}_{company_name}/, filename:
     cv_{name}_{job_title}_{company_name}.pdf). There was previously no
     structured company/job-title input to this pipeline at all (only
     the free-text job_description) -- this is a genuinely new input the
     caller (gencv/cli.py) must now supply.
  2. build_resume_yaml() call renamed to build_cv_yaml() (name only).
  3. render_pdf() call updated: pdf_path is now computed up front via
     build_cv_pdf_path() and passed explicitly (render_pdf()'s pdf_path
     parameter is now required, no more settings.yaml fallback).
  4. name resolution moved earlier in the function body (before the
     build_cv_pdf_path() call) since the PDF filename now needs it --
     previously `name` was only consumed much later, inside
     build_resume_yaml().
- v10 (2026-08-16): Wired resume_builder.fetch_domain_job_function_tags()
  into the pipeline between rewrite_and_confirm() and
  draft_and_confirm_summary().
- v9 (2026-08-16): Wired in claim_ranker.rank_and_narrow_claims().
- v8 (2026-08-16): build_resume_yaml() call updated to pass the required
  job_description parameter.
- v7 (2026-08-16): Simplified to match resume_builder.py v12's
  build_education_entries() signature change.
- v6 (2026-08-16): [SUPERSEDED by v7].
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
5. claim_ranker.rank_and_narrow_claims() -- 同一 project 底下相關 claim
   超過上限（2）時，narrow 到上限，優先依 tag Jaccard 多樣性選擇
6. education_selector.select_relevant_degrees() -- 決定 education section 顯示哪些學位
7. experience_rewriter.rewrite_and_confirm() -- 融合 STAR + 貼近 JD 用語，逐條人工確認
   （依 project 分組，同組多筆 claim 時要求能力面向多樣化改寫）
8. resume_builder.build_education_entries() -- 組出 education entry 清單
9. resume_builder.fetch_domain_job_function_tags() -- 撈出 domain/job_function
   tag，作為 summary 的輔助訊號
10. summary_writer.draft_and_confirm_summary() -- 總結整份履歷，含 domain/
    job_function tag 輔助訊號，單次確認
11. resume_builder.build_cv_pdf_path() -- 依 name/job_title/company_name
    算出這次 PDF 的完整輸出路徑（v11 新增）
12. resume_builder.build_cv_yaml() -- 組裝履歷內容（skills section 僅含
    tag_category == "skill" 的 tag）
13. resume_builder.render_pdf() -- 呼叫 rendercv CLI 排版編譯成 PDF，
    輸出到步驟11算好的路徑
14. application_log.log_generated_output() -- 記錄這次投遞
"""

import pathlib
import tempfile

from gencv.experience_matcher import select_claims
from gencv.claim_ranker import rank_and_narrow_claims
from gencv.experience_rewriter import rewrite_and_confirm
from gencv.resume_builder import build_cv_yaml, render_pdf
from core.db import get_session, JobORM

def generate_cv(
    job_id: str,
    designated_name: str | None = None,
    designated_headline: str | None = None,
) -> pathlib.Path:
    """JD -> PDF 單一入口。

    company_name / job_title (v11, new, REQUIRED): 用於組出這次 PDF 的
    輸出路徑（見 resume_builder.build_cv_pdf_path()）--
    {OUTPUT_DIR}/{job_title}_{company_name}/cv_{name}_{job_title}_{company_name}.pdf。

    Returns:
        PDF 路徑
    """
    with get_session() as session:
        job = session.get(JobORM, job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found.")
        job_description = job.job_description
        company_name = job.company_name
        job_title = job.job_title

    headline = job_title if designated_headline is None else designated_headline
        
    selected_claim_ids = select_claims(job_description)

    if not selected_claim_ids:
        raise ValueError("篩選後沒有任何經歷通過（可能被治理層擋下，或 LLM 判定皆不相關）。")

    selected_claim_ids = rank_and_narrow_claims(selected_claim_ids, job_description) 

    final_descriptions = rewrite_and_confirm(selected_claim_ids, job_description)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cv_yaml_path = pathlib.Path(tmp_dir) / "cv.yaml"
        pdf_path = build_cv_yaml(
            company_name=company_name,
            job_title=job_title,
            job_description=job_description,
            claim_ids=selected_claim_ids,
            output_path=str(cv_yaml_path),
            designated_name=designated_name,
            headline=headline,
            description_overrides=final_descriptions,          
        )
        render_pdf(cv_yaml_path=cv_yaml_path, pdf_path=pdf_path)

    return pdf_path
