"""Tool: Tavily web search fallback when RAG context is insufficient."""

from __future__ import annotations

import logging
import os

from tavily import TavilyClient

from agent.state import WebHit

logger = logging.getLogger(__name__)


class TavilySearchTool:
    """Thin wrapper over the Tavily client.

    If ``TAVILY_API_KEY`` is unset the tool stays ``available = False`` and
    ``search`` returns an empty list — the orchestrator then skips the web action.
    """

    def __init__(self) -> None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            self.available = False
            self._client: TavilyClient | None = None
            logger.warning("TAVILY_API_KEY not set — web search disabled")
            return
        self.available = True
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[WebHit]:
        if not self.available or self._client is None:
            return []
        try:
            resp = self._client.search(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 — network/3rd-party failures are non-fatal
            logger.warning("Tavily search failed: %s", exc)
            return []

        hits: list[WebHit] = []
        for item in resp.get("results", []):
            hits.append(
                WebHit(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                )
            )
        return hits
