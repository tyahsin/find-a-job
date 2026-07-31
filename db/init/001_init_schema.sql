-- 001_init_schema.sql
-- Source of Truth schema for experiences_db

-- 1. 啟用 pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 經驗背景層
CREATE TABLE entity ( /* tyahsin: how about role based ? */
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company TEXT, /* CV 呈現與時間軸排序的基本資訊 */
    industry TEXT, /* 對應 JD 產業要求，做初步篩選 */
    location TEXT, /* 對應 JD 地區要求，也方便你篩選跨國經驗 */
    title TEXT, /* 決定敘事語氣與角色定位 */
    department TEXT, /* 幫助判斷 Role Scope，釐清協作範圍 */
    start_date DATE, /* 排序、判斷經驗新舊、計算年資 */
    end_date DATE NULL /* 搭配 start_date 計算任職時長；若仍在職可設為 NULL */
);

-- 3. 專案層（新增：問題應對應到具體專案，而非整個職位期間）
CREATE TABLE project (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- entity_id UUID REFERENCES entity(entity_id), /* 對應哪個公司/職位 */
    project_name TEXT, /* 專案名稱，方便識別與檢索 */
    project_start_date DATE, /* 專案起始時間，可能短於整個職位任期 */
    project_end_date DATE NULL, /* 專案結束時間，若仍在進行可設為 NULL */
    project_context TEXT /* 補充這個專案在什麼情境下產生 */
);

-- 4. 問題層
CREATE TABLE problem (
    problem_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES project(project_id), /* 對應哪個專案，而非公司或職位本身 */
    description TEXT, /* 問題描述：應然與實然的落差 */
    stakeholder_impact TEXT, /* 這個落差影響誰、影響多大 */
    baseline_metric TEXT, /* 問題發生前的現況指標 */
    evidence_level TEXT /* L1-L4，避免把自認為的問題包裝成正式任務 */
);

-- 5. 行動層
CREATE TABLE action (
    action_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID REFERENCES problem(problem_id), /* 對應解決哪個問題 */
    description TEXT, /* 具體做了什麼 */
    role_scope TEXT /* 主導/協同/支援 */
);


-- 6. 結果/聲稱層
CREATE TABLE result_claim (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_id UUID REFERENCES action(action_id), /* 對應哪個行動產生此結果 */
    result_statement TEXT, /* 結果敘述，CV bullet 的核心內容來源 */
    value_type TEXT, /* 商業價值類型：營收/成本/時間/風險/流程/團隊賦能 */
    metric TEXT, /* 量化數字 */
    metric_confidence TEXT, /* 數字精確度：精確值/估算/方向性 */
    skill_tags TEXT[], /* 對應技能關鍵字 */
    domain_tags TEXT[], /* 對應產業 */
    job_function_tags TEXT[] /* 對應職能 */
); /* tyahsin: 這裡所有的欄位都需要再過一遍 */

-- 7. 證據層
CREATE TABLE evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES result_claim(claim_id), /* 對應哪條結果聲稱 */
    evidence_type TEXT, /* 原始檔案/結構化紀錄/自述/推論 */
    evidence_location TEXT, /* 檔案路徑或連結 */
    verified_date DATE /* 最後確認證據有效的日期 */
);

-- 8. 治理層
CREATE TABLE governance_metadata (
    metadata_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES result_claim(claim_id), /* 對應哪條 claim */
    support_level TEXT, /* 綜合證據強度分級 L1-L4 */
    -- confidence FLOAT, /* 主觀信心分數 */
    use_policy TEXT, /* 使用限制，如僅限資深職缺/特定產業 */
    notes_for_llm TEXT, /* 給生成模型的操作指示 */
    allowed_for_generation BOOLEAN DEFAULT TRUE /* 是否允許進入生成候選池，治理層硬性開關 */
);

-- 9. 向量索引層（供 pgvector 使用）
CREATE TABLE claim_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES result_claim(claim_id),
    entity_id UUID,
    project_id UUID,
    problem_id UUID,
    action_id UUID,
    support_level TEXT,
    allowed_for_generation BOOLEAN,
    domain_tags TEXT[],
    skill_tags TEXT[],
    job_function_tags TEXT[],
    content TEXT, /* 用於嵌入的文字內容 */
    embedding VECTOR(1536) /* pgvector 相似度搜尋的核心資料 */
);

CREATE INDEX ON claim_chunks USING hnsw (embedding vector_cosine_ops);

-- 10. 輸出追蹤層
CREATE TABLE generated_outputs (
    output_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_reference TEXT,
    used_claim_ids UUID[],
    output_type TEXT,
    generated_date TIMESTAMP DEFAULT now(),
    human_review_status TEXT
);