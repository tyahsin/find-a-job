"""
onboard/prompts.py
Prompt templates used by the onboard CLI flows to draft:
- position.responsibility (from a source file OR from a spoken/typed summary)
- project.project_context (from a short user-provided description)
- action.description (what was done)
- claim.description (what result followed from the action -- deliberately
  scoped AWAY from action's territory, see RESULT_CLAIM_SYSTEM_PROMPT below)

Design principle (matches the project's governance rules): the LLM only
rephrases/organizes what the user already provided. It must NOT invent
responsibilities, scope, dates, or achievements that aren't present in the
input. All drafts are shown to the user for confirmation/edit before being
persisted (precompute-once, human-reviewed).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared system-level guardrails
# ---------------------------------------------------------------------------

_BASE_SYSTEM_RULES = """你是一個協助整理個人職涯與學業經歷的寫作助手。你的任務是把使用者提供的原始素材，
整理成一段簡潔、清楚、專業的中文摘要文字。

嚴格規則（必須遵守）：
1. 不得新增任何使用者提供的素材中沒有出現過的具體事實、數字、職責範圍或成就。
2. 不得將不確定或模糊的描述，改寫成語氣更確定、更誇大的說法。
3. 不得推測、延伸使用者沒有明確提到的技能、職責或貢獻範圍。
4. 只能做以下事情：語句重組、用詞潤飾、去除贅字、統一語氣、提升可讀性。
5. 若原始素材內容過於簡短或模糊，摘要也應維持同等程度的簡短與模糊，不要自行「補完」。

輸出格式：只輸出摘要本身的純文字，不要加任何前言、標題、或說明性文字。
"""

# ---------------------------------------------------------------------------
# position.responsibility
# ---------------------------------------------------------------------------

RESPONSIBILITY_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
這段摘要的用途：描述使用者在某個職位期間的職責範圍（position.responsibility）。
摘要應該是客觀的職責描述，不是成就敘述（成就會在其他地方單獨處理），
結果必須以英文呈現。
"""
# 長度建議在2到4句話之間。
# """


def build_responsibility_prompt_from_source(source_text: str) -> str:
    """
    Use when the user provided a source file (JD, offer letter, etc.)
    via responsibility_source. The LLM summarizes the ACTUAL document
    content -- it must not add responsibilities beyond what's written.
    """
    return f"""以下是使用者提供的原始文件內容（可能是職位說明書、offer letter等），
請將其中關於「職責範圍」的部分整理成一段簡潔摘要：

---
{source_text}
---

請輸出摘要（純文字，不要加任何前言）："""


def build_responsibility_prompt_from_description(user_description: str) -> str:
    """
    Use when the user has no source file and instead typed/spoke a rough
    description of their responsibilities. The LLM only polishes wording;
    it must not add scope beyond what the user described.
    """
    return f"""使用者用自己的話描述了這個職位的職責範圍，內容可能較口語或簡短，
請將其整理成一段簡潔、professional的摘要，但不要新增使用者沒有提到的內容：

---
{user_description}
---
結果必須以英文呈現。
請輸出摘要（純文字，不要加任何前言）："""


# ---------------------------------------------------------------------------
# project.project_context
# ---------------------------------------------------------------------------

PROJECT_CONTEXT_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
這段摘要的用途：描述一個專案發生的背景與情境（project.project_context），
說明這個專案是在什麼狀況下產生的、屬於什麼類型的專案。
這不是成果敘述（成果會透過problem/action/result_claim另外處理），
結果必須以英文呈現。
"""
# 長度建議在1到3句話之間簡潔呈現。
# """


def build_project_context_prompt(user_description: str) -> str:
    """
    Use for drafting project.project_context from the user's short,
    free-text description of the project's background.
    """
    return f"""使用者簡述了一個專案的背景/情境，請將其整理成一段簡潔的專案背景敘述：

---
{user_description}
---

請輸出摘要（純文字，不要加任何前言）："""


# ---------------------------------------------------------------------------
# problem.description
# ---------------------------------------------------------------------------

PROBLEM_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
這段摘要的用途：描述這個專案中遇到的具體問題（problem.description），
說明「當時面臨什麼狀況/困難/缺口」，不要描述任何解決方式（那是 action 的職責）。
結果必須以英文呈現。
"""
# 長度建議在1到2句話之間。
# """

def build_problem_prompt(user_description: str) -> str:
    return f"""使用者描述了這個專案中遇到的問題，請整理成一段簡潔的問題敘述：

---
{user_description}
---

請輸出摘要（純文字，不要加任何前言）："""


# ---------------------------------------------------------------------------
# action.description
# ---------------------------------------------------------------------------

ACTION_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
這段摘要的用途：描述使用者針對前述問題「做了什麼」（action.description），
只描述具體採取的行動、方法、或介入方式，不要描述行動帶來的結果或影響
（那是 result_claim 的職責）。
結果必須以英文呈現。
"""
# 長度建議在1到2句話之間。
# """


def build_action_prompt(user_description: str) -> str:
    return f"""使用者描述了針對這個問題採取的行動，請整理成一段簡潔的行動敘述：

---
{user_description}
---

請輸出摘要（純文字，不要加任何前言）："""


# ---------------------------------------------------------------------------
# claim.description
# ---------------------------------------------------------------------------
# NOTE: previously this prompt's Chinese wording ("請描述這個結果") overlapped
# semantically with action's wording, causing user confusion about which
# layer they were filling in. This version explicitly draws the boundary:
# description = what happened AFTER the action, not the action itself.

RESULT_CLAIM_SYSTEM_PROMPT = _BASE_SYSTEM_RULES + """
這段摘要的用途：描述一個具體行動（action）所帶來的「結果」（claim.description）。
只描述結果本身 -- 例如數據變化、產出、影響、效率提升、被驗收/採用的狀態等。
不要重複描述「做了什麼」（那是 action 的職責），只描述「做了之後發生了什麼」。
結果必須以英文呈現。
"""
# 長度建議在1到2句話之間，愈精簡愈好。
# """


def build_result_claim_prompt(user_description: str) -> str:
    return f"""使用者描述了這個行動帶來的結果，請整理成一段簡潔的結果敘述
（只描述結果，不要重複行動本身）：

---
{user_description}
---

請輸出摘要（純文字，不要加任何前言）："""


# ---------------------------------------------------------------------------
# evidence_type -> evidence_level mapping (Problem 3)
# ---------------------------------------------------------------------------
# Rationale: evidence strength is a function of evidence TYPE, not a
# separate subjective judgment call. Letting the user pick L1-L4 freely
# leads to inconsistent self-grading over time (the same kind of evidence
# might be graded L2 today and L3 next month). Instead, the CLI presents a
# fixed menu of evidence types, and evidence_level is derived automatically.
#
# L1 = strongest (verifiable, third-party or system-of-record evidence)
# L4 = weakest (unverifiable, self-reported recollection)

EVIDENCE_TYPE_TO_LEVEL: dict[str, str] = {
    "正式文件（合約/正式報告/系統截圖）": "L1",
    "第三方驗證（主管評語/客戶回饋/正式考核）": "L1",
    "數據紀錄（分析報表/系統 log/dashboard）": "L2",
    "團隊內部文件（會議紀錄/email/Slack 對話）": "L2",
    "口頭轉述/回憶（無書面紀錄，僅個人記憶）": "L3",
    "推測/估算（合理推論但無直接證據支持）": "L4",
}


# ---------------------------------------------------------------------------
# tag generation (Problem 1 & 2 & 3 from Aug 2026 review)
# ---------------------------------------------------------------------------
# Design: tags are NOT typed by the user. Once problem/action/claim/evidence
# are all filled in for a given add-claim run, the LLM reads the FULL
# accumulated text of that run (already held in memory -- the user never
# re-pastes anything) and proposes a set of tags. The user then
# confirms/edits the proposed list (not a yes/no gate -- tagging itself is
# a mandatory step of add-claim, only the exact tag content is adjustable).

TAG_SYSTEM_PROMPT = """你是一個協助整理職涯經歷標籤的助手。根據使用者提供的問題、行動、結果、證據描述，
以及這筆經歷所連結的背景資訊（公司/職位或學校/學位），產生一組精簡的標籤，用於未來 CV 生成時比對職缺說明（JD）。

嚴格規則（必須遵守）：
1. 標籤必須是輸入內容（含背景資訊）中確實體現的技能、領域、或職能，不得憑空發明未出現過的能力。
   背景資訊只能用來輔助判斷 domain（例如公司產業）或 job_function（例如職稱），
   不能單獨依據背景資訊生成 skill 標籤（skill 必須有 problem/action/result 文字佐證）。
2. 每個標籤只能屬於以下三種類別之一：skill（技能，例如 Python、資料分析）、
   domain（領域，例如 金融、教育科技）、job_function（職能，例如 產品管理、資料工程）。
3. 標籤命名要精簡、通用、可重複使用（例如用「資料分析」而非「用 pandas 做資料清理跟分析」）。
4. 最多產生8個標籤，寧可少不要浮誇。

輸出格式：每行一個標籤，格式為 "標籤名稱 | 類別"（類別只能是 skill/domain/job_function）。
不要加任何前言、說明文字、或多餘的標點符號。
"""


def build_tag_prompt(
    problem_description: str,
    action_description: str,
    description: str,
    evidence_type: str,
    static_context: str = "",
) -> str:
    """
    Builds the tag-generation prompt directly from the in-memory content
    already collected during this add-claim run -- problem/action/claim/
    evidence text is passed in as function arguments, not re-typed by the
    user.

    `static_context` carries the claim's linked static (relatively
    unchanging) data -- company/position or school/degree info -- resolved
    upstream from the claim -> action -> problem -> project -> position/
    degree -> company/school chain. Including this lets the LLM infer
    domain/job_function tags that are implied by *where* the work happened
    (e.g. "金融" from company industry, "資料工程" from job title) even when
    that context isn't explicitly repeated in the problem/action/result text.
    """
    context_block = f"\n背景資訊（公司/職位或學校/學位）：{static_context}\n" if static_context else ""
    return f"""以下是這一筆經歷紀錄的完整內容：
{context_block}
問題：{problem_description}
行動：{action_description}
結果：{description}
證據類型：{evidence_type}

請根據以上內容（包含背景資訊）產生標籤（格式：標籤名稱 | 類別）："""


def parse_tag_response(raw_text: str) -> list[tuple[str, str]]:
    """
    Parses lines like "Python | skill" into (tag_name, tag_category) tuples.
    Skips malformed lines rather than raising, since this is LLM output that
    the user will review/edit anyway before it's persisted.
    """
    valid_categories = {"skill", "domain", "job_function"}
    results: list[tuple[str, str]] = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name_part, _, category_part = line.partition("|")
        name = name_part.strip()
        category = category_part.strip().lower()
        if name and category in valid_categories:
            results.append((name, category))
    return results


# ---------------------------------------------------------------------------
# fixed-choice menus (Aug 2026 update: role_scope / value_type /
# metric_confidence should be selected from a list, not free-typed)
# ---------------------------------------------------------------------------

ROLE_SCOPE_OPTIONS: list[str] = [
    "Led",
    "Collaborated",
    "Supported",
]

# NOTE (Aug 2026): these options must come strictly from the classification
# already defined for their related columns in prompts.py / schema notes --
# not invented additions. value_type mirrors the exact three categories
# given in the original description prompt wording (效率提升/成本節省/
# 品質改善); metric_confidence mirrors the exact three categories given in
# the original metric_confidence prompt wording (精確/估算/大致印象).

VALUE_TYPE_OPTIONS: list[str] = [
    "效率提升",
    "成本節省",
    "品質改善",
]

# metric_type reuses the same three categories plus "其他" for metrics that
# don't cleanly fit (this is exactly the gap that motivated splitting
# metrics out into their own claim_metric table).
METRIC_TYPE_OPTIONS: list[str] = VALUE_TYPE_OPTIONS + ["其他"]

METRIC_CONFIDENCE_OPTIONS: list[str] = [
    "精確",
    "估算",
    "大致印象",
]


# ---------------------------------------------------------------------------
# claim_metric extraction (Aug 2026 update)
# ---------------------------------------------------------------------------
# LLM only EXTRACTS metrics that are explicitly present in description
# text -- it must not estimate, invent, or self-report a confidence score
# (LLM self-reported confidence is known to be poorly calibrated). The user
# always reviews/edits the extracted draft, and metric_confidence is always
# a manual human selection from METRIC_CONFIDENCE_OPTIONS, never LLM output.

METRIC_EXTRACTION_SYSTEM_PROMPT = """你的任務是從一段結果敘述（description）中，抽取所有可量化的指標。

嚴格規則（必須遵守）：
1. 只抽取文字中明確出現的數字/量化描述，不得推測、估算、或發明文字中沒寫出來的數據。
2. 每個指標輸出三個欄位：類型、標籤、數值。
   - 類型只能是「效率提升」「成本節省」「品質改善」「其他」四種之一，抓不進前三類就用「其他」。
   - 標籤：這個數字實際代表什麼（例如「處理時間」「客戶滿意度」），用精簡詞語描述。
   - 數值：把原文中的數字/描述完整保留（例如「從5天降到2天」「約15%」）。
3. 不要輸出信心程度，信心程度由使用者自行判斷，與你無關。
4. 如果這段文字沒有任何可量化指標，回傳空白（不輸出任何行）。

輸出格式：每行一組，用 | 分隔：類型|標籤|數值
不要加任何前言、說明文字、或多餘的標點符號。
"""


def build_metric_extraction_prompt(description: str) -> str:
    return f"""以下是這筆 claim 的結果敘述：

{description}

請抽取其中所有可量化的指標（格式：類型|標籤|數值）："""


def parse_metric_response(raw_text: str) -> list[tuple[str, str, str]]:
    """
    Parses lines like "效率提升|處理時間|從5天降到2天" into
    (metric_type, metric_label, metric_value) tuples. Skips malformed lines
    rather than raising, since this is an LLM draft the user reviews anyway.
    """
    valid_types = {"效率提升", "成本節省", "品質改善", "其他"}
    results: list[tuple[str, str, str]] = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or line.count("|") != 2:
            continue
        metric_type, metric_label, metric_value = (part.strip() for part in line.split("|"))
        if metric_type not in valid_types or not metric_label or not metric_value:
            continue
        results.append((metric_type, metric_label, metric_value))
    return results
