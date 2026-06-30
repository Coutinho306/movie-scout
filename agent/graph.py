"""LangGraph wiring: orchestrator + RAG/web workers + synthesize."""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config import AgentSettings
from agent.nodes.orchestrator import orchestrator_node
from agent.nodes.rag import rag_node
from agent.nodes.synthesize import synthesize_node
from agent.nodes.web import web_node
from agent.state import AgentState


def build_graph(settings: AgentSettings | None = None) -> CompiledStateGraph:
    """Build and compile the 4-node Movie Scout agent graph."""
    settings = settings or AgentSettings()
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("orchestrator", lambda s: orchestrator_node(s, settings))
    graph.add_node("rag_agent", lambda s: rag_node(s, settings))
    graph.add_node("web_agent", lambda s: web_node(s, settings))
    graph.add_node("synthesize", lambda s: synthesize_node(s, settings))

    graph.set_entry_point("orchestrator")

    def route(state: AgentState) -> str:
        # Hard safety stop on the turn cap regardless of the LLM's action.
        if state["orchestrator_turns"] >= settings.max_orchestrator_turns:
            return "synthesize"
        last_action = state["plan"][-1] if state.get("plan") else "rag"
        return {
            "rag": "rag_agent",
            "web": "web_agent",
            "synthesize": "synthesize",
        }.get(last_action, "synthesize")

    graph.add_conditional_edges(
        "orchestrator",
        route,
        {
            "rag_agent": "rag_agent",
            "web_agent": "web_agent",
            "synthesize": "synthesize",
        },
    )

    graph.add_edge("rag_agent", "orchestrator")
    graph.add_edge("web_agent", "orchestrator")
    graph.add_edge("synthesize", END)

    return graph.compile()
