/*用途：目前 001_init_schema.sql 只在 claim_chunks.embedding 上建了 HNSW 索引，其他表目前沒有額外索引。如果之後查詢量變大，可以補上：

claim_inventory 的 domain_tags、skill_tags 用 GIN 索引加速陣列查詢。

governance_metadata.allowed_for_generation 建索引，因為這是 hybrid retrieval 每次都會用到的過濾條件。

補上 CHECK constraint，例如限制 support_level 只能是 'L1'/'L2'/'L3'/'L4'。*/