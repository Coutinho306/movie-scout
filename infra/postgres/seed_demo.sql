-- Opt-in demo data (spec 0009). Not run by container bootstrap — apply manually:
--   docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < infra/postgres/seed_demo.sql
-- Inserts 50 fake runs (spread over the last 14 days) + 30 feedback rows so the
-- Grafana dashboard renders populated without running the agent 50 times.
-- Deterministic: values derive from the row index, no random().

INSERT INTO agent_runs (
    run_id, ts, user_query, final_answer,
    latency_ms, cost_usd, tool_calls, rag_calls, web_calls,
    orchestrator_turns, model, prompt_variant, citations
)
SELECT
    -- deterministic UUID from the index
    ('00000000-0000-4000-8000-' || lpad(g::text, 12, '0'))::uuid,
    now() - (g * interval '6 hours'),
    'demo query ' || g,
    'Recommended film for demo run ' || g,
    200 + (g % 40) * 25,                       -- latency_ms 200..1175
    0.0005 + (g % 10) * 0.0003,                -- cost_usd
    1 + (g % 3),                               -- tool_calls
    1 + (g % 2),                               -- rag_calls
    (g % 2),                                   -- web_calls
    1 + (g % 3),                               -- orchestrator_turns
    'gpt-4o-mini',
    CASE WHEN g % 2 = 0 THEN 'baseline' ELSE 'rewrite' END,
    jsonb_build_array(
        jsonb_build_object(
            'tmdb_id', 600 + (g % 10),         -- 10 distinct ids for the top-N panel
            'title', 'Demo Movie ' || (g % 10),
            'year', 1980 + (g % 10),
            'why_for_you', 'Matches your taste (demo).',
            'provider_hint', NULL
        )
    )
FROM generate_series(1, 50) AS g
ON CONFLICT (run_id) DO NOTHING;

INSERT INTO agent_feedback (run_id, ts, rating, comment)
SELECT
    ('00000000-0000-4000-8000-' || lpad(g::text, 12, '0'))::uuid,
    now() - (g * interval '6 hours') + interval '2 minutes',
    CASE WHEN g % 4 = 0 THEN 'down' ELSE 'up' END,   -- ~25% thumbs down
    CASE WHEN g % 4 = 0 THEN 'Not quite my taste (demo).' ELSE NULL END
FROM generate_series(1, 30) AS g;
