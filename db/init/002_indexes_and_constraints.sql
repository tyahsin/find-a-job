/*用途：目前 001_init_schema.sql 只在 claim_chunks.embedding 上建了 HNSW 索引，其他表目前沒有額外索引。如果之後查詢量變大，可以補上：

claim_inventory 的 domain_tags、skill_tags 用 GIN 索引加速陣列查詢。

governance_metadata.allowed_for_generation 建索引，因為這是 hybrid retrieval 每次都會用到的過濾條件。

補上 CHECK constraint，例如限制 support_level 只能是 'L1'/'L2'/'L3'/'L4'。*/

-- Enforce at most one default identity at the database level (partial
-- unique index -- only rows where is_default = TRUE participate in the
-- uniqueness check, so any number of FALSE rows are allowed).
CREATE UNIQUE INDEX identity_only_one_default
ON identity (is_default)
WHERE is_default = TRUE;

-- ---------------------------------------------------------------------------
-- apply/ 模組索引（2026-08-19 新增，配合 001_init_schema.sql 的 job/application 表）
-- ---------------------------------------------------------------------------

-- current_stage 是 CLI 查詢/dashboard 最常用的過濾條件（例如「列出所有
-- Interviewing 狀態的投遞」），建索引加速。
CREATE INDEX idx_application_current_stage
ON application (current_stage);

-- job_id 外鍵查詢（例如「這個職缺投過幾次」），建索引加速 JOIN。
CREATE INDEX idx_application_job_id
ON application (job_id);
