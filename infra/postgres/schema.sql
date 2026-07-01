-- Movie Scout monitoring schema (spec 0009).
-- Canonical DDL. Copied verbatim into init/01-schema.sql for container bootstrap.

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id             UUID PRIMARY KEY,
    ts                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_query         TEXT,
    final_answer       TEXT,
    latency_ms         NUMERIC,
    cost_usd           NUMERIC,
    tool_calls         INT,
    rag_calls          INT,
    web_calls          INT,
    orchestrator_turns INT,
    model              TEXT,
    prompt_variant     TEXT,
    citations          JSONB
);

CREATE TABLE IF NOT EXISTS agent_feedback (
    id      BIGSERIAL PRIMARY KEY,
    run_id  UUID REFERENCES agent_runs(run_id),
    ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    rating  TEXT CHECK (rating IN ('up', 'down')),
    comment TEXT
);

CREATE INDEX IF NOT EXISTS agent_runs_ts_idx ON agent_runs (ts);
CREATE INDEX IF NOT EXISTS agent_feedback_run_id_idx ON agent_feedback (run_id);
