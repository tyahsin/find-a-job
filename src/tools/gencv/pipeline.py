"""
gencv/pipeline.py
"""

import pathlib
import tempfile

from gencv.claim_selector import select_claims
from gencv.claim_ranker import narrow_all_claims
from gencv.claim_rewriter import rewrite_and_confirm
from gencv.resume_builder import build_cv_yaml, render_pdf
from core.db import get_session, JobORM


def generate_cv(
    job_id: str,
    designated_name: str | None = None,
    designated_headline: str | None = None,
    log_path: str | None = None,
) -> pathlib.Path:
    """JD -> PDF 單一入口。

    company_name / job_title (v11, new, REQUIRED): 用於組出這次 PDF 的
    輸出路徑（見 resume_builder.build_cv_pdf_path()）--
    {OUTPUT_DIR}/{job_title}_{company_name}/cv_{name}_{job_title}_{company_name}.pdf。

    log_path: 選填，轉交給 narrow_all_claims -- 每個 group 只會被 append
    最終確認過的 KEEP/MERGE/DROP 決策，不含中間過程。

    Returns:
        PDF 路徑
    """
    # 只在真正需要查 DB 的這一段開 session：撈 job 本身、以及 select_claims
    # 內部那次一次性的 SQL（已經把 notes_for_llm/org_name 等後面全部階段
    # 需要的欄位都撈出來了）。narrow_all_claims 和 rewrite_and_confirm 完全
    # 不碰 DB，而且中間有多輪人工confirm，沒必要讓 session 一直開著等人。
    with get_session() as session:
        job = session.get(JobORM, job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found.")
        job_description = job.job_description
        company_name = job.company_name
        job_title = job.job_title

        selected = select_claims(session, job_description)

    if not selected["work_track"] and not selected["academic_track"]:
        raise ValueError("篩選後沒有任何經歷通過（可能被治理層擋下，或 LLM 判定皆不相關）。")

    headline = job_title if designated_headline is None else designated_headline

    # Stage 2 + 3: 依 position/project 做 MECE 的 KEEP/MERGE/DROP narrowing。
    # 回傳 (items, contexts) -- contexts 沿用 select_claims 撈出來的同一份
    # 資料，rewrite_and_confirm 不用再查一次 DB。
    final_items, contexts = narrow_all_claims(selected, job_description, log_path=log_path)

    if not final_items:
        raise ValueError("MECE narrowing 後沒有任何 item 留下，無法產生履歷內容。")

    # 逐筆 fuse + JD-tailor + 人工 confirm，回傳的每個 item 都保留原本的
    # group_key/group_label/track/kind/reasons，多了一個 "description"。
    rewritten = rewrite_and_confirm(final_items, contexts, job_description)

    # resume_builder.build_cv_yaml 還沒更新成吃完整的 item list -- 暫時把
    # rewritten 攤平成舊版認得的 claim_ids + description_overrides，等
    # resume_builder.py 真正支援 merge/reasons 之後再換成整包 item 傳進去。
    # 一個 merge item 底下的每個 claim_id 暫時都指到同一句融合後的
    # description，這是攤平成舊格式時無法避免的近似。
    claim_ids = [cid for item in rewritten for cid in item["claim_ids"]]
    description_overrides = {
        cid: item["description"] for item in rewritten for cid in item["claim_ids"]
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        cv_yaml_path = pathlib.Path(tmp_dir) / "cv.yaml"
        pdf_path = build_cv_yaml(
            company_name=company_name,
            job_title=job_title,
            job_description=job_description,
            claim_ids=claim_ids,
            description_overrides=description_overrides,
            output_path=str(cv_yaml_path),
            designated_name=designated_name,
            headline=headline,
        )
        render_pdf(cv_yaml_path=cv_yaml_path, pdf_path=pdf_path)

    return pdf_path