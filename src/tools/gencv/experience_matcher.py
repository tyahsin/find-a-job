"""
experience_matcher.py
JD -> 判斷哪些過往經歷跟這個職缺相關,決定要納入這次履歷的候選清單。
含共用查詢工具:一次性把 claim -> action -> problem -> project -> position/company
(或 degree/school)的完整背景資訊撈出來。

core/db.py 的 ORM models 只用 Column(ForeignKey(...)) 建立外鍵約束,沒有定義
relationship(),因此無法用 claim.action.problem... 這種屬性鏈存取,必須明確寫 JOIN 查詢。

流程(computationally light 設計,依候選池規模 ~250-300 筆決定):
  Step 1: 治理層硬性過濾 -- GovernanceORM.allowed_for_generation = True 才進入候選池
  Step 2: 單次批次 LLM 判斷 -- 把全部候選經歷的摘要一次塞進同一個 prompt,
          只呼叫一次 LLM,請它直接回傳相關的 claim_id 清單。
          取代「每筆經歷各呼叫一次 LLM」的作法;若未來候選池遠超 1000+ 筆規模,
          可考慮在此步驟之前先加一層向量相似度粗篩(用 claim_chunks 的 embedding)。
"""

import re
import uuid
from sqlalchemy import text
from core.db import get_session, GovernanceORM
from core.llm_client import get_llm_client

RELEVANCE_SYSTEM_PROMPT = """你是一個履歷篩選助手。以下會給你一份職缺敘述(Job Description),
以及一份求職者過往經歷清單(每筆有編號、背景、問題、行動、成果,以及該筆經歷的使用規範)。
請判斷每一筆經歷是否與該職缺相關、值得放進針對這個職缺客製化的履歷裡。

若某筆經歷的使用規範(use_policy)明確排除此職缺類型,一律視為不相關。

輸出格式:只輸出相關經歷的編號,一行一個,不要其他文字、不要編號以外的說明。
如果沒有任何一筆相關,輸出 NONE。"""

_EXPERIENCE_CONTEXT_SQL = """
    SELECT
        claim.claim_id,
        claim.description AS claim_description,
        action.description AS action_description,
        action.role_scope,
        problem.description AS problem_description,
        problem.stakeholder_impact,
        project.project_name,
        project.project_context,
        position.title AS position_title,
        position.department,
        position.start_date AS position_start,
        position.end_date AS position_end,
        company.company_name,
        degree.degree_name,
        degree.subject AS degree_subject,
        degree.start_date AS degree_start,
        degree.end_date AS degree_end,
        school.school_name
    FROM claim
    JOIN action ON claim.action_id = action.action_id
    JOIN problem ON action.problem_id = problem.problem_id
    JOIN project ON problem.project_id = project.project_id
    LEFT JOIN position ON project.position_id = position.position_id
    LEFT JOIN company ON position.company_id = company.company_id
    LEFT JOIN degree ON project.degree_id = degree.degree_id
    LEFT JOIN school ON degree.school_id = school.school_id
    WHERE claim.claim_id = ANY(:claim_ids)
"""


def to_uuid_list(claim_ids: list) -> list[uuid.UUID]:
    """claim_id 對外一律用字串(UUID 文字)傳遞,餵進 SQLAlchemy PGUUID 欄位過濾前
    需轉成 uuid.UUID 物件。"""
    return [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]


def fetch_experience_contexts(session, claim_ids: list[str]) -> dict[str, dict]:
    """回傳 {claim_id字串: context_dict} 映射,一次 JOIN 查詢取得完整背景資訊。

    is_work_track=True 代表這筆經歷掛在 position(工作經歷)底下,
    False 代表掛在 degree(學業經歷)底下(project.position_id / degree_id 二擇一)。
    """
    if not claim_ids:
        return {}

    rows = session.execute(
        text(_EXPERIENCE_CONTEXT_SQL), {"claim_ids": to_uuid_list(claim_ids)}
    ).mappings().all()

    contexts = {}
    for row in rows:
        claim_id = str(row["claim_id"])
        is_work_track = row["position_title"] is not None
        contexts[claim_id] = {
            "claim_description": row["claim_description"],
            "action_description": row["action_description"],
            "role_scope": row["role_scope"],
            "problem_description": row["problem_description"],
            "stakeholder_impact": row["stakeholder_impact"],
            "project_name": row["project_name"],
            "project_context": row["project_context"],
            "is_work_track": is_work_track,
            "org_name": row["company_name"] if is_work_track else row["school_name"],
            "title": row["position_title"] if is_work_track else row["degree_name"],
            "start_date": row["position_start"] if is_work_track else row["degree_start"],
            "end_date": row["position_end"] if is_work_track else row["degree_end"],
        }
    return contexts


def _allowed_governance(session) -> dict[str, GovernanceORM]:
    """Step 1: 只取 allowed_for_generation = True 的治理紀錄,claim_id -> GovernanceORM 映射。"""
    rows = session.query(GovernanceORM).filter(
        GovernanceORM.allowed_for_generation.is_(True)
    ).all()
    return {str(g.claim_id): g for g in rows}


def _build_batch_prompt(
    candidate_ids: list[str],
    contexts: dict[str, dict],
    governances: dict[str, GovernanceORM],
    job_description: str,
) -> tuple[str, dict[int, str]]:
    """組出單一批次 prompt,把所有候選經歷編號列出。回傳 (prompt, {編號: claim_id})。"""
    index_to_claim_id: dict[int, str] = {}
    lines = []
    for i, claim_id in enumerate(candidate_ids, 1):
        ctx = contexts.get(claim_id)
        governance = governances.get(claim_id)
        if not ctx or not governance:
            continue
        index_to_claim_id[i] = claim_id
        org_context = f"{ctx['org_name']} - {ctx['title']}" if ctx["org_name"] else "無背景資訊"
        lines.append(
            f"[{i}] 背景: {org_context}\n"
            f"    問題: {ctx['problem_description']}\n"
            f"    行動: {ctx['action_description']}\n"
            f"    成果: {ctx['claim_description']}\n"
            f"    使用規範: use_policy={governance.use_policy or '無'}, "
            f"notes_for_llm={governance.notes_for_llm or '無'}"
        )

    experiences_block = "\n\n".join(lines)
    user_prompt = f"""# 職缺敘述(Job Description)
{job_description}

# 過往經歷清單(共 {len(index_to_claim_id)} 筆)
{experiences_block}"""

    return user_prompt, index_to_claim_id


def _parse_relevant_indices(raw_response: str) -> set[int]:
    """從 LLM 回應中抓出所有數字編號,容忍多餘文字/markdown 包裝。"""
    if "NONE" in raw_response.upper() and not re.search(r"\d", raw_response):
        return set()
    return {int(n) for n in re.findall(r"\d+", raw_response)}


def select_claims(job_description: str) -> list[str]:
    """主函式:單次批次 LLM 呼叫,回傳最終應納入這次履歷的 claim_id(UUID字串)清單。"""
    with get_session() as session:
        governances = _allowed_governance(session)
        if not governances:
            return []

        candidate_ids = list(governances.keys())
        contexts = fetch_experience_contexts(session, candidate_ids)

        user_prompt, index_to_claim_id = _build_batch_prompt(
            candidate_ids, contexts, governances, job_description
        )

        if not index_to_claim_id:
            return []

        response = get_llm_client().generate_text(
            system_prompt=RELEVANCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1000,
        )

        relevant_indices = _parse_relevant_indices(response.text)
        selected_ids = [
            index_to_claim_id[i] for i in relevant_indices if i in index_to_claim_id
        ]

    return selected_ids
