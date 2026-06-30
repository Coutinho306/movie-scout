"""LangGraph node implementations for the Movie Scout agent."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt template (``name`` without extension) from agent/prompts/."""
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
