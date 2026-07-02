"""LangGraph wiring: orchestrator + RAG/web workers + synthesize."""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config import AgentSettings
from agent.nodes.orchestrator import orchestrator_node
from agent.nodes.rag import rag_node
from agent.nodes.rewrite import rewrite_node
from agent.nodes.synthesize import synthesize_inform_node, synthesize_node
from agent.nodes.web import web_node
from agent.state import AgentState


def _route(state: AgentState, max_turns: int) -> str:
    """Pick the next node from orchestrator state. Module-level so it's unit-testable."""
    # inform queries end at the prose node; everything else at the recommender.
    terminal = "synthesize_inform" if state.get("intent") == "inform" else "synthesize"
    # Hard safety stop on the turn cap regardless of the LLM's action.
    if state["orchestrator_turns"] >= max_turns:
        return terminal
    plan = state.get("plan", [])
    last_action = plan[-1] if plan else "rag"
    # Repeat-guard: don't burn a turn re-running the same worker back-to-back
    # (mirrors the orchestrator prompt rule). The turn cap stays the hard stop.
    if len(plan) >= 2 and plan[-1] == plan[-2] and last_action in {"rag", "web"}:
        return terminal
    return {
        "rag": "rewrite",  # rewrite once, then hand off to the RAG worker
        "web": "web_agent",
        "synthesize": terminal,
    }.get(last_action, terminal)


def build_graph(settings: AgentSettings | None = None) -> CompiledStateGraph:
    """Build and compile the 4-node Movie Scout agent graph."""
    settings = settings or AgentSettings()
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("orchestrator", lambda s: orchestrator_node(s, settings))
    graph.add_node("rewrite", lambda s: rewrite_node(s, settings))
    graph.add_node("rag_agent", lambda s: rag_node(s, settings))
    graph.add_node("web_agent", lambda s: web_node(s, settings))
    graph.add_node("synthesize", lambda s: synthesize_node(s, settings))
    graph.add_node("synthesize_inform", lambda s: synthesize_inform_node(s, settings))

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        lambda s: _route(s, settings.max_orchestrator_turns),
        {
            "rewrite": "rewrite",
            "web_agent": "web_agent",
            "synthesize": "synthesize",
            "synthesize_inform": "synthesize_inform",
        },
    )

    graph.add_edge("rewrite", "rag_agent")
    graph.add_edge("rag_agent", "orchestrator")
    graph.add_edge("web_agent", "orchestrator")
    graph.add_edge("synthesize", END)
    graph.add_edge("synthesize_inform", END)

    return graph.compile()
