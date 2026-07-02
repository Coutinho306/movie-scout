.PHONY: up down ingest eval logs build

# Bring up the runtime stack (postgres, grafana, api, frontend; Qdrant is cloud).
up:
	docker compose up -d

# Stop and remove the stack (keeps named volumes).
down:
	docker compose down

# One-shot ingestion (profile-gated so `up` never runs it). Pass ARGS=... to append.
ingest:
	docker compose --profile ingest run --rm ingest $(ARGS)

# Retrieval + LLM eval grids on the host (needs Qdrant populated + OPENAI_API_KEY).
eval:
	uv run python3 -m eval.cli all

# Tail logs for all services.
logs:
	docker compose logs -f

# Build all images.
build:
	docker compose --profile ingest build
