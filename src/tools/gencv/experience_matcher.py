"""
experience_matcher.py
Version: v3 (2026-08-16)

Changelog:
- v3 (2026-08-16): Added degree.degree_id -> context["degree_id"] to
  _EXPERIENCE_CONTEXT_SQL / fetch_experience_contexts(). This was missing
  from v2 -- resume_builder.build_project_entries() needs each
  education-track claim's degree_id to collect degree_ids_with_projects
  (the set build_education_entries() uses to filter relevant_degrees down
  to only degrees with at least one selected project underneath them).
  Without this column, ctx.get("degree_id") was always None and the
  education filtering logic in resume_builder.py silently produced an
  empty education section every time. Also added project.project_id ->
  context["project_id"] for symmetry (not currently consumed elsewhere,
  but the query already joins through project).
- v2 (2026-08-12): Fixed two gaps between this file's actual behavior and the
  intended design recorded in project notes:
  1. Governance filtering (_resolve_candidate_ids, replaces the old
     _allowed_governance): previously only claims WITH a governance row AND
     allowed_for_generation = True were kept, silently excluding every claim
     that predates the governance layer (no governance row at all). Now the
     candidate pool = all claims, MINUS claims with an explicit
     allowed_for_generation = False. Claims with no governance row default to
     allowed, matching the schema's allowed_for_generation BOOLEAN DEFAULT
     TRUE design intent.
  2. Tag signal (fetch_tags_for_claims, new): the batch relevance prompt now
     includes each candidate's claim_tag-linked tags (skill/domain/
     job_function), with an explicit system-prompt instruction that tags are
     an important relevance signal. Previously the LLM only saw free-text
     problem/action/result descriptions and had no visibility into
     user-confirmed tags.
- v1 (2026-08-10, inferred): Initial batch-LLM relevance matcher.

JD -> 判斷哪些過往經歷跟這個職缺相關，決定要納入這次履歷的候選清單。
含共用查詢工具：一次性把 claim -> action -> problem -> project -> position/company
（或 degree/school）的完整背景資訊撈出來。

core/db.py 的 ORM models 只用 Column(ForeignKey(...)) 建立外鍵約束，沒有定義
relationship()，因此無法用 claim.action.problem... 這種屬性鏈存取，必須明確寫 JOIN 查詢。

流程（computationally light 設計，依候選池規模 ~250-300 筆決定）：
Step 1: 治理層過濾 -- 沒有 governance 紀錄的 claim 預設視為允許（對應 schema
allowed_for_generation BOOLEAN DEFAULT TRUE 的設計精神），只有明確
allowed_for_generation = False 的 claim 才被排除。
Step 2: 單次批次 LLM 判斷 -- 把全部候選經歷的摘要（含 tag 標籤）一次塞進同一個
prompt，只呼叫一次 LLM，請它直接回傳相關的 claim_id 清單。
取代「每筆經歷各呼叫一次 LLM」的作法；若未來候選池遠超 1000+ 筆規模，
可考慮在此步驟之前先加一層向量相似度粗篩（用 claim_chunks 的 embedding）。
"""

import re
import uuid
from collections import defaultdict

from sqlalchemy import text
from core.db import get_session, GovernanceORM, ClaimORM, ClaimTagORM, TagORM
from core.llm_client import get_llm_client

RELEVANCE_SYSTEM_PROMPT = """你是一個履歷篩選助手。以下會給你一份職缺敘述（Job Description），
以及一份求職者過往經歷清單（每筆有編號、背景、問題、行動、成果、標籤，以及該筆經歷的使用規範）。
請判斷每一筆經歷是否與該職缺相關、值得放進針對這個職缺客製化的履歷裡。

標籤（tags）是重要的判斷訊號 -- 標籤代表使用者已手動確認這筆經歷具備的技能/領域/職能，
即使成果描述文字本身沒有明確關鍵字，標籤命中職缺需求時仍應視為相關。

若某筆經歷的使用規範（use_policy）明確排除此職缺類型，一律視為不相關。

輸出格式：只輸出相關經歷的編號，一行一個，不要其他文字、不要編號以外的說明。
如果沒有任何一筆相關，輸出 NONE。"""

_EXPERIENCE_CONTEXT_SQL = """
SELECT
    claim.claim_id,
    claim.description AS claim_description,
    action.description AS action_description,
    action.role_scope,
    problem.description AS problem_description,
    problem.stakeholder_impact,
    project.project_id,
    project.project_name,
    project.project_context,
    position.title AS position_title,
    position.department,
    position.start_date AS position_start,
    position.end_date AS position_end,
    company.company_name,
    degree.degree_id,
    degree.degree_level,
    degree.degree_name,
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
    """claim_id 對外一律用字串（UUID 文字）傳遞，餵進 SQLAlchemy PGUUID 欄位過濾前
    需轉成 uuid.UUID 物件。"""
    return [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]


def fetch_experience_contexts(session, claim_ids: list[str]) -> dict[str, dict]:
    """回傳 {claim_id字串: context_dict} 映射，一次 JOIN 查詢取得完整背景資訊。

    is_work_track=True 代表這筆經歷掛在 position（工作經歷）底下，
    False 代表掛在 degree（學業經歷）底下（project.position_id / degree_id 二擇一）。

    degree_id（v3 新增）：is_work_track=False 時，這筆 claim 所屬的 degree_id
    （字串形式）。resume_builder.build_project_entries() 用它來收集
    degree_ids_with_projects，交給 build_education_entries() 篩選學位清單。
    is_work_track=True 時此欄位為 None。
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
            "project_id": str(row["project_id"]) if row["project_id"] else None,
            "project_name": row["project_name"],
            "project_context": row["project_context"],
            "is_work_track": is_work_track,
            "org_name": row["company_name"] if is_work_track else row["school_name"],
            "title": row["position_title"] if is_work_track else row["degree_name"],
            "start_date": row["position_start"] if is_work_track else row["degree_start"],
            "end_date": row["position_end"] if is_work_track else row["degree_end"],
            "degree_id": str(row["degree_id"]) if (not is_work_track and row["degree_id"]) else None,
        }
    return contexts


def fetch_tags_for_claims(session, claim_ids: list[str]) -> dict[str, list[str]]:
    """claim_id -> [tag_name, ...] 映射，一次 JOIN 查詢取得每筆 claim 的所有標籤。

    (v2) 用於把使用者手動確認過的標籤塞進批次判斷 prompt，作為比成果描述文字
    更明確的相關性訊號。
    """
    if not claim_ids:
        return {}

    rows = (
        session.query(ClaimTagORM.claim_id, TagORM.tag_name)
        .join(TagORM, ClaimTagORM.tag_id == TagORM.tag_id)
        .filter(ClaimTagORM.claim_id.in_(to_uuid_list(claim_ids)))
        .all()
    )

    tags_map: dict[str, list[str]] = defaultdict(list)
    for claim_id, tag_name in rows:
        tags_map[str(claim_id)].append(tag_name)
    return tags_map


def _resolve_candidate_ids(session) -> tuple[list[str], dict[str, GovernanceORM]]:
    """Step 1: 治理層過濾。

    候選清單 = 全部 claim_id，排除掉「有 governance 紀錄且明確
    allowed_for_generation = False」的 claim。沒有 governance 紀錄的 claim
    （例如治理層加入之前建立的舊 claim）預設視為允許，對應 schema
    allowed_for_generation BOOLEAN DEFAULT TRUE 的設計精神。

    Returns:
        candidate_ids: 允許進入候選池的 claim_id 字串清單。
        governance_map: claim_id -> GovernanceORM，只包含「確實有 governance
        紀錄」的 claim（用於後續組 prompt 時讀取 use_policy/notes_for_llm；
        沒有紀錄的 claim 在 governance_map 裡不存在，呼叫端應以「無」處理）。
    """
    all_claim_ids = [str(row[0]) for row in session.query(ClaimORM.claim_id).all()]

    governance_rows = session.query(GovernanceORM).all()
    governance_map = {str(g.claim_id): g for g in governance_rows}

    candidate_ids = [
        cid for cid in all_claim_ids
        if governance_map.get(cid) is None or governance_map[cid].allowed_for_generation is not False
    ]

    return candidate_ids, governance_map


def _build_batch_prompt(
    candidate_ids: list[str],
    contexts: dict[str, dict],
    governances: dict[str, GovernanceORM],
    tags_map: dict[str, list[str]],
    job_description: str,
) -> tuple[str, dict[int, str]]:
    """組出單一批次 prompt，把所有候選經歷編號列出（含標籤）。回傳 (prompt, {編號: claim_id})。"""
    index_to_claim_id: dict[int, str] = {}
    lines = []
    for i, claim_id in enumerate(candidate_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        index_to_claim_id[i] = claim_id

        governance = governances.get(claim_id)
        use_policy = governance.use_policy if governance else None
        notes_for_llm = governance.notes_for_llm if governance else None

        tags = tags_map.get(claim_id, [])
        tags_line = ", ".join(tags) if tags else "無"

        org_context = f"{ctx['org_name']} - {ctx['title']}" if ctx["org_name"] else "無背景資訊"
        lines.append(
            f"[{i}] 背景: {org_context}\n"
            f"    問題: {ctx['problem_description']}\n"
            f"    行動: {ctx['action_description']}\n"
            f"    成果: {ctx['claim_description']}\n"
            f"    標籤: {tags_line}\n"
            f"    使用規範: use_policy={use_policy or '無'}, "
            f"notes_for_llm={notes_for_llm or '無'}"
        )
    experiences_block = "\n\n".join(lines)
    user_prompt = f"""# 職缺敘述（Job Description）
{job_description}

# 過往經歷清單（共 {len(index_to_claim_id)} 筆）
{experiences_block}"""

    return user_prompt, index_to_claim_id


def _parse_relevant_indices(raw_response: str) -> set[int]:
    """從 LLM 回應中抓出所有數字編號，容忍多餘文字/markdown 包裝。"""
    if "NONE" in raw_response.upper() and not re.search(r"\d", raw_response):
        return set()
    return {int(n) for n in re.findall(r"\d+", raw_response)}


def select_claims(job_description: str) -> list[str]:
    """主函式：單次批次 LLM 呼叫，回傳最終應納入這次履歷的 claim_id（UUID字串）清單。"""
    with get_session() as session:
        candidate_ids, governances = _resolve_candidate_ids(session)
        if not candidate_ids:
            return []

        contexts = fetch_experience_contexts(session, candidate_ids)
        tags_map = fetch_tags_for_claims(session, candidate_ids)

        user_prompt, index_to_claim_id = _build_batch_prompt(
            candidate_ids, contexts, governances, tags_map, job_description
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
