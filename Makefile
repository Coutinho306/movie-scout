.PHONY: up down ingest eval logs build snapshot restore-seed lint

# Bring up the runtime stack (postgres, grafana, api, frontend, qdrant).
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

# Create local Qdrant snapshots of both collections -> data/snapshots/.
# Requires the qdrant service to be running: docker compose up -d qdrant
snapshot:
	uv run python3 scripts/snapshot_qdrant.py

# Restore both collections from local snapshot files into a running local Qdrant.
# Keyless — no QDRANT_API_KEY needed.
# Usage:
#   docker compose up -d qdrant
#   make restore-seed
restore-seed:
	uv run python3 scripts/restore_qdrant.py

lint:
	uv run ruff check .
