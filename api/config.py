"""API-time settings — DB URL, CORS origins, port. Loaded from .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str | None = None  # postgresql://…; None disables persistence
    allowed_origins: str = "http://localhost:8501"  # comma-separated
    port: int = 8000
    rate_limit: str = "10/minute"  # slowapi limit string applied to /ask

    def origins_list(self) -> list[str]:
        """Split ALLOWED_ORIGINS into a clean list for the CORS middleware."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
