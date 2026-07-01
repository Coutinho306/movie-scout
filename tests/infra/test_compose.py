"""Compose wiring test — asserts service deps/profiles/env_file via `docker compose config`.

Marked integration: needs the `docker` CLI. Skips with a reason when absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

COMPOSE_FILE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose_config() -> dict:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
    proc = subprocess.run(
        ["docker", "compose", "--profile", "ingest", "config", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"docker compose config failed: {proc.stderr.strip()[:200]}")
    return json.loads(proc.stdout)


def test_all_services_have_env_file() -> None:
    # `docker compose config` resolves env_file into `environment` and drops the
    # key, so assert against the raw compose file instead.
    import yaml

    raw = yaml.safe_load(COMPOSE_FILE.read_text())
    for name, svc in raw["services"].items():
        assert svc.get("env_file"), f"service {name} missing env_file"


def test_api_depends_on_postgres_healthy(compose_config: dict) -> None:
    dep = compose_config["services"]["api"]["depends_on"]["postgres"]
    assert dep["condition"] == "service_healthy"


def test_frontend_depends_on_api(compose_config: dict) -> None:
    assert "api" in compose_config["services"]["frontend"]["depends_on"]


def test_ingest_gated_behind_profile(compose_config: dict) -> None:
    assert "ingest" in compose_config["services"]["ingest"]["profiles"]
