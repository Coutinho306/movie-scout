"""Web worker node: a ReAct agent over a single Tavily search tool."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.config import AgentSettings
from agent.cost import usage_from_messages
from agent.nodes import load_prompt
from agent.state import AgentState
from agent.tools.web_search import TavilySearchTool

logger = logging.getLogger(__name__)


def _build_web_tools(collected: list[dict], tavily: TavilySearchTool, max_results: int) -> list:
    """Build a single ``tavily_search`` tool bound to a run-local collector."""

    @tool
    def tavily_search(query: str) -> list[dict]:
        """Search the web for current movie info: reviews, streaming availability, recent releases. Returns snippets with url, title, content."""
        hits = tavily.search(query, max_results=max_results)
        dicts = [h.model_dump() for h in hits]
        seen = {d["url"] for d in collected}
        for d in dicts:
            if d["url"] not in seen:
                collected.append(d)
                seen.add(d["url"])
        return dicts

    return [tavily_search]


def build_web_agent(settings: AgentSettings, collected: list[dict], tavily: TavilySearchTool):
    """Construct a ReAct agent whose tool appends web hits to ``collected``."""
    llm = ChatOpenAI(model=settings.model_agent, temperature=0)
    tools = _build_web_tools(collected, tavily, settings.tavily_max_results)
    return create_react_agent(llm, tools=tools, prompt=load_prompt("web_system"))


def web_node(state: AgentState, settings: AgentSettings) -> dict:
    """Run the web ReAct agent, merge captured hits into state, track usage.

    If Tavily is unavailable (no API key), records a no-op web call.
    """
    tavily = TavilySearchTool()
    if not tavily.available:
        logger.warning("Web node invoked but Tavily unavailable — skipping search")
        return {"web_calls": state.get("web_calls", 0) + 1}

    collected: list[dict] = list(state.get("web_hits", []))
    agent = build_web_agent(settings, collected, tavily)

    query = state.get("rewritten_query") or state["user_query"]
    result = agent.invoke({"messages": [HumanMessage(content=query)]})

    messages = result.get("messages", [])
    tokens, cost = usage_from_messages(messages, settings.model_agent)

    return {
        "web_hits": collected,
        "web_calls": state.get("web_calls", 0) + 1,
        "token_count": state.get("token_count", 0) + tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
    }
