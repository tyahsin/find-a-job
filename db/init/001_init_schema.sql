-- 001_init_schema.sql
-- Source of Truth schema for experiences_db

CREATE EXTENSION IF NOT EXISTS vector;

-- 個人資訊層
CREATE TABLE identity (
    identity_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             TEXT NOT NULL,
    location         TEXT NOT NULL,
    visa_status      TEXT NOT NULL,  
    email            TEXT NOT NULL,
    phone            TEXT NOT NULL,
    is_default       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE network (
    network_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network         TEXT NOT NULL,        -- 例如 "GitHub", "LinkedIn", "Behance"
    username        TEXT NOT NULL,
    target_position TEXT [],     -- 這個連結適用的工種，例如 "Software Engineer", "Designer"
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 公司層
CREATE TABLE company (
    company_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name TEXT,
    location TEXT
);

-- 職位層
CREATE TABLE position (
    position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES company(company_id),
    title TEXT,
    department TEXT,
    responsibility TEXT, /* precompute：定版簡述，人工或LLM生成皆存在此欄位 */
    responsibility_source TEXT DEFAULT NULL, /* 原始資料連結/路徑；NULL代表無原文，簡述為LLM生成 */
    start_date DATE,
    end_date DATE NULL
);

-- 學校層
CREATE TABLE school (
    school_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    school_name TEXT,
    location TEXT
);

-- 學業層
CREATE TABLE degree (
    degree_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    school_id UUID REFERENCES school(school_id),
    degree_level TEXT, /* MA, MS, PhD, Short Program, etc. */
    degree_type TEXT, /* Major, Minor, Concentration, etc. */
    degree_name TEXT, 
    degree_description TEXT, /* tyahsin: precompute：定版簡述，人工或LLM生成皆存在此欄位 */
    degree_description_source TEXT DEFAULT NULL, /* 原始資料連結/路徑；NULL代表無原文，簡述為LLM生成 */
    -- degree_honors TEXT [], /* tyahsin: cum laude, magna cum laude, summa cum laude, etc. */
    -- degree_awards TEXT [], /* tyahsin: scholarships, fellowships, etc. */
    start_date DATE,
    end_date DATE NULL
);

-- 專案層（可掛在position或degree底下，二擇一）
CREATE TABLE project (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id UUID REFERENCES position(position_id) NULL,
    degree_id UUID REFERENCES degree(degree_id) NULL,
    project_name TEXT,
    start_date DATE,
    end_date DATE NULL,
    project_context TEXT, /* tyahsin: the background of the project; why was it initiated amd why it's passed onto me */
    CHECK (
    (position_id IS NOT NULL AND degree_id IS NULL) OR
    (position_id IS NULL AND degree_id IS NOT NULL)
    )
);

-- 問題層
CREATE TABLE problem (
    problem_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES project(project_id), /* 對應哪個專案，而非公司或職位本身 */
    description TEXT, /* 問題描述：應然與實然的落差 */
    stakeholder_impact TEXT, /* 這個落差影響誰、影響多大 */
    baseline_metric TEXT /* 問題發生前的現況指標 */ /* tyahsin: should this mirror some columns in claim table? */
);

-- 行動層
CREATE TABLE action (
    action_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID REFERENCES problem(problem_id), /* 對應解決哪個問題 */
    description TEXT, /* 具體做了什麼 */
    role_scope TEXT /* 主導/協同/支援 (Led/Collaborated/Supported) */
);

-- 結果/聲稱層 (Claim)
CREATE TABLE claim (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_id UUID REFERENCES action(action_id), /* 對應哪個行動產生此結果 */
    description TEXT /* 結果敘述，CV bullet 的核心內容來源（原 result_statement） */
);

-- 量化指標層 (Claim Metric) -- 一個 claim 可能有多個指標，且不是所有指標
-- 都能歸類到單一 value_type，因此拆成獨立子表 (one-to-many)。
-- metric_confidence 一律由使用者人工判定，不採用 LLM 自報信心（LLM 自報
-- 信心分數普遍缺乏校準，不可靠）。
CREATE TABLE claim_metric (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claim(claim_id), /* 對應哪條結果聲稱 */
    metric_type TEXT, /* 效率提升 / 成本節省 / 品質改善 / 其他 */
    metric_label TEXT, /* 這個數字代表什麼，例如「處理時間」「客戶滿意度」 */ /*tyahsin: pretty redundant*/
    metric_value TEXT, /* 數值本身，含單位/描述，例如「從5天降到2天」「約15%」 */
    metric_confidence TEXT /* 精確 / 估算 / 大致印象 -- 一律人工判定 */
);

-- 證據層
-- NOTE: 一個 claim 可能有多筆證據（例如一筆 L1 直接檔案 + 一筆 L3 補充說明），
-- 本表結構原本就支援一對多（claim_id 無 UNIQUE 限制），無需改動。
CREATE TABLE evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claim(claim_id), /* 對應哪條結果聲稱 */
    evidence_type TEXT, /* 原始檔案/結構化紀錄/自述/推論 */
    evidence_location TEXT, /* 檔案路徑或連結 */
    evidence_level TEXT /* L1-L4，避免把自認為的問題包裝成正式任務 */
);

-- 治理層
CREATE TABLE governance (
    metadata_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claim(claim_id), /* 對應哪條 claim */
    support_level TEXT, /* 綜合證據強度分級 L1-L4 */
    use_policy TEXT DEFAULT NULL, /* 使用限制，如僅限資深職缺/特定產業 */ -- 限制範圍（能不能用）
    notes_for_llm TEXT DEFAULT NULL, /* 給生成模型的操作指示 */ -- 操作指示（怎麼寫）
    allowed_for_generation BOOLEAN DEFAULT TRUE /* 是否允許進入生成候選池，治理層硬性開關 */
);

-- 受控詞彙表：所有 tag 必須從這裡選，新增需人工確認
CREATE TABLE tag (
    tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tag_name TEXT UNIQUE NOT NULL, -- 標準寫法，唯一
    tag_category TEXT NOT NULL CHECK (
    tag_category IN ('skill', 'domain', 'job_function')
    ), -- 可轉移的能力, 產業領域知識, 職能類別
    aliases TEXT[] DEFAULT '{}', -- 同義詞/其他措辭，供 fuzzy match 比對用
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP
);

-- claim 與 tag 的多對多關聯
CREATE TABLE claim_tag (
    claim_id UUID REFERENCES claim(claim_id),
    tag_id UUID REFERENCES tag(tag_id),
    PRIMARY KEY (claim_id, tag_id)
);

-- 向量索引層（供 pgvector 使用）
CREATE TABLE claim_chunks ( /* 這段文字語意上跟 JD 有多相似 */
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES claim(claim_id),
    support_level TEXT, /* 一筆claim可能有多筆evidence; 用support_level做加總平均 */
    content TEXT, /* 這段文字語意上跟 JD 有多相似 */
    embedding VECTOR(1536) /* pgvector 相似度搜尋的核心資料 */ /* tyahsin: why 1536? */
); /* tyahsin: what is the best structure for a pgvector-oriened table? */
/* tyahsin: 建議做法是 onboard 時仍生成一組「基礎/廣義」tags 作為預設候選（方便一般檢索與瀏覽），
CV generation 時再讓 LLM 針對特定 JD 做「措辭轉譯」而非從零生成，這樣既保留了你要的靈活性，
也避免每次都要從無到有分析整條 claim 全文的成本浪費*/

CREATE INDEX ON claim_chunks USING hnsw (embedding vector_cosine_ops);

-- 輸出追蹤層
CREATE TABLE generated_outputs (
    output_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_reference TEXT,
    used_claim_ids UUID[],
    output_type TEXT,
    generated_date TIMESTAMP DEFAULT now(),
    human_review_status BOOLEAN /* tyahsin: how about just boolean to make sure it's either human reviewed or not? */
);
