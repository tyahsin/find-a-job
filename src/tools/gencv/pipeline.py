"""
gencv/pipeline.py
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
