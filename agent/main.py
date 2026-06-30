"""LangGraph agent entrypoint for Movie Scout."""

from __future__ import annotations

import json
import logging
import time

from agent.config import AgentSettings
from agent.graph import build_graph
from agent.state import AgentRunResult, AgentState, RecItem

logger = logging.getLogger(__name__)


def run(user_query: str, settings: AgentSettings | None = None) -> AgentRunResult:
    """Run the agent end-to-end for a single query and return a structured result."""
    settings = settings or AgentSettings()
    graph = build_graph(settings)

    start = time.time()
    initial_state: AgentState = {
        "user_query": user_query,
        "rewritten_query": None,
        "plan": [],
        "rag_hits": [],
        "web_hits": [],
        "messages": [],
        "final_answer": None,
        "recs": [],
        "orchestrator_turns": 0,
        "rag_calls": 0,
        "web_calls": 0,
        "cost_usd": 0.0,
        "token_count": 0,
    }

    result = graph.invoke(initial_state)
    latency_ms = (time.time() - start) * 1000

    citations: list[RecItem] = []
    for item in result.get("recs", []):
        try:
            citations.append(RecItem(**item))
        except Exception:  # noqa: BLE001 — skip malformed rec
            continue

    log_line = {
        "step": "agent_run",
        "tokens": result.get("token_count", 0),
        "cost_usd": round(result.get("cost_usd", 0.0), 6),
        "orchestrator_turns": result.get("orchestrator_turns", 0),
        "rag_calls": result.get("rag_calls", 0),
        "web_calls": result.get("web_calls", 0),
        "latency_ms": round(latency_ms, 1),
    }
    logger.info(json.dumps(log_line))

    return AgentRunResult(
        final_answer=result.get("final_answer") or "No recommendation generated.",
        citations=citations,
        tool_calls=result.get("rag_calls", 0) + result.get("web_calls", 0),
        latency_ms=latency_ms,
        cost_usd=result.get("cost_usd", 0.0),
        orchestrator_turns=result.get("orchestrator_turns", 0),
        rag_calls=result.get("rag_calls", 0),
        web_calls=result.get("web_calls", 0),
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    query = " ".join(sys.argv[1:]) or "recommend something slow and meditative"
    run_result = run(query)
    print(run_result.final_answer)
