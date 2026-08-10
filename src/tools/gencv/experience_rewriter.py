"""
experience_rewriter.py
把 experience_matcher.py 篩選出的經歷,用 LLM 做「輕度改寫」,讓措辭貼近目標職缺(JD)用語。

安全設計原則(對應 governance 治理層精神,避免 overclaiming):
  1. 數字絕對不讓 LLM 碰。metric 數值一律機械式附加在句子後面,LLM 不參與生成。
  2. 尊重 governance.notes_for_llm(給生成模型的操作指示),當作改寫時的硬性約束。
  3. 只換用字、不重組句子結構或誇大成果。
  4. 每一條都逐句要求人工確認(Enter 接受 / n 打回原句),不做批次靜默接受。

GROUND RULES(REWRITE_SYSTEM_PROMPT)目前寫死在本檔案的常數字串裡,若未來想不改
程式碼就能調整規則,可考慮移到 core/config.py 或獨立設定檔。

用法(由 gencv/pipeline.py 呼叫,不單獨對外暴露):
    from gencv.experience_rewriter import rewrite_and_confirm
    rewritten = rewrite_and_confirm(selected_claim_ids, job_description)
"""

from core.db import get_session, GovernanceORM
from core.llm_client import get_llm_client
from gencv.experience_matcher import fetch_experience_contexts

REWRITE_SYSTEM_PROMPT = """你是一個履歷措辭優化助手。你的任務是把求職者的一句真實工作成果描述,
換句話說讓它更貼近指定職缺(JD)的用語與強調重點,但必須遵守以下規則:

1. 絕對不能新增、誇大、或改變任何事實、數字、成果幅度 -- 只能換用字遣詞。
2. 不能重組句子的行動主體或責任歸屬(例如不能把「協助」改成「主導」)。
3. 必須遵守「操作指示」欄位裡的具體限制。
4. 只輸出改寫後的句子本身,不要其他說明文字、不要引號、不要編號。
5. 如果這句話已經很貼近 JD 用語,或找不到更好的換法,原樣輸出即可,不要為改而改。"""

REWRITE_USER_PROMPT_TEMPLATE = """# 職缺敘述(Job Description)
{job_description}

# 原始成果描述
{claim_description}

# 這筆經歷的操作指示(必須遵守,若為「無」則無特別限制)
{notes_for_llm}

請輸出改寫後的句子:"""


def _fetch_governance_notes(session, claim_ids: list[str]) -> dict[str, str]:
    """claim_id -> notes_for_llm 映射,用於改寫時的硬性約束。"""
    if not claim_ids:
        return {}
    import uuid
    uuid_ids = [uuid.UUID(cid) if isinstance(cid, str) else cid for cid in claim_ids]
    rows = session.query(GovernanceORM).filter(GovernanceORM.claim_id.in_(uuid_ids)).all()
    return {str(g.claim_id): (g.notes_for_llm or "無") for g in rows}


def _rewrite_one(claim_description: str, job_description: str, notes_for_llm: str) -> str:
    user_prompt = REWRITE_USER_PROMPT_TEMPLATE.format(
        job_description=job_description,
        claim_description=claim_description,
        notes_for_llm=notes_for_llm,
    )
    response = get_llm_client().generate_text(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=200,
    )
    return response.text.strip()


def _confirm_one(index: int, total: int, org_label: str, title_label: str,
                  original: str, rewritten: str) -> str:
    """逐條顯示原句 vs 改寫,讓使用者決定用哪一版。回傳最終採用的文字。"""
    print(f"\n[{index}/{total}] [{org_label} - {title_label}]")
    print(f"  原句: {original}")
    print(f"  改寫: {rewritten}")

    if original.strip() == rewritten.strip():
        print("  (改寫結果與原句相同,直接採用)")
        return original

    choice = input("  接受此改寫嗎?(Enter接受 / n還原原句): ").strip().lower()
    if choice == "n":
        print("  → 已還原為原句")
        return original
    print("  → 已採用改寫版")
    return rewritten


def rewrite_and_confirm(claim_ids: list[str], job_description: str) -> dict[str, str]:
    """對每一筆 claim 做輕度改寫,逐條要求人工確認,回傳最終要用的敘述文字。"""
    with get_session() as session:
        contexts = fetch_experience_contexts(session, claim_ids)
        notes_map = _fetch_governance_notes(session, claim_ids)

    print(f"\n=== 開始逐條改寫 {len(claim_ids)} 筆經歷的措辭,貼近此次職缺用語 ===")

    final_descriptions: dict[str, str] = {}
    for i, claim_id in enumerate(claim_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue

        original = ctx["claim_description"]
        notes_for_llm = notes_map.get(claim_id, "無")
        rewritten = _rewrite_one(original, job_description, notes_for_llm)

        org_label = ctx["org_name"] or "無背景"
        title_label = ctx["title"] or ""
        final_descriptions[claim_id] = _confirm_one(
            i, len(claim_ids), org_label, title_label, original, rewritten
        )

    print("=" * 50)
    return final_descriptions
