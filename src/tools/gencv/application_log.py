"""
application_log.py
對應 schema 裡的 generated_outputs 表 -- 記錄每次投遞產出的履歷,
把用了哪些經歷、投給哪個職缺寫下來,避免之後回頭查不到。
"""

from core.db import get_session, GeneratedOutputORM
from gencv.experience_matcher import to_uuid_list


def log_generated_output(
    job_description: str,
    used_claim_ids: list[str],
    output_type: str = "pdf",
) -> str:
    """寫入一筆投遞紀錄,human_review_status 預設 False,代表「已生成但尚未人工複核」。"""
    with get_session() as session:
        record = GeneratedOutputORM(
            jd_reference=job_description,
            used_claim_ids=to_uuid_list(used_claim_ids),
            output_type=output_type,
            human_review_status=False,
        )
        session.add(record)
        session.flush()
        output_id = str(record.output_id)

    return output_id


def mark_reviewed(output_id: str) -> None:
    """人工複核完成後呼叫,把 human_review_status 標記為 True。"""
    import uuid
    with get_session() as session:
        record = session.query(GeneratedOutputORM).filter(
            GeneratedOutputORM.output_id == uuid.UUID(output_id)
        ).one()
        record.human_review_status = True
