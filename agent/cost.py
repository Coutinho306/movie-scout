"""Token + USD cost accounting from LangChain message usage_metadata."""

from __future__ import annotations

from langchain_core.messages import BaseMessage

# USD per 1M tokens (input, output). Covers the default + likely swap-in models.
# Unknown models fall back to gpt-4o-mini pricing to avoid crashing on cost math.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
}
_DEFAULT = _PRICING["gpt-4o-mini"]


def usage_from_message(message: BaseMessage, model: str) -> tuple[int, float]:
    """Return ``(total_tokens, cost_usd)`` for one LLM response message."""
    meta = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(meta.get("input_tokens", 0))
    output_tokens = int(meta.get("output_tokens", 0))
    total = int(meta.get("total_tokens", input_tokens + output_tokens))

    in_price, out_price = _PRICING.get(model, _DEFAULT)
    cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return total, cost


def usage_from_messages(messages: list[BaseMessage], model: str) -> tuple[int, float]:
    """Aggregate token + cost across a list of messages (e.g. a ReAct run)."""
    total_tokens = 0
    total_cost = 0.0
    for msg in messages:
        if getattr(msg, "usage_metadata", None):
            tok, cost = usage_from_message(msg, model)
            total_tokens += tok
            total_cost += cost
    return total_tokens, total_cost
