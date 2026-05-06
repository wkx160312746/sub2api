CREATE TABLE IF NOT EXISTS image_tasks (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    request_body JSONB NOT NULL DEFAULT '{}'::jsonb,
    api_key_id BIGINT REFERENCES api_keys(id) ON DELETE SET NULL,
    user_id BIGINT REFERENCES users(id) ON DELETE SET NULL,
    group_id BIGINT REFERENCES groups(id) ON DELETE SET NULL,
    account_id BIGINT REFERENCES accounts(id) ON DELETE SET NULL,
    model TEXT NOT NULL DEFAULT '',
    model_alias TEXT NOT NULL DEFAULT '',
    image_tier TEXT NOT NULL DEFAULT '',
    aspect_ratio TEXT NOT NULL DEFAULT '',
    target_width INTEGER NOT NULL DEFAULT 0,
    target_height INTEGER NOT NULL DEFAULT 0,
    upstream_size TEXT NOT NULL DEFAULT '',
    output_format TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT 'application/json',
    channel_mapped_model TEXT NOT NULL DEFAULT '',
    channel_usage_fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    request_payload_hash TEXT NOT NULL DEFAULT '',
    user_agent TEXT NOT NULL DEFAULT '',
    ip_address TEXT NOT NULL DEFAULT '',
    inbound_endpoint TEXT NOT NULL DEFAULT '',
    result_json JSONB,
    error_code TEXT,
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_image_tasks_status_created
    ON image_tasks (status, created_at);

CREATE INDEX IF NOT EXISTS idx_image_tasks_api_key_created
    ON image_tasks (api_key_id, created_at DESC);

