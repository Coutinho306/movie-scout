"""Streamlit frontend for Movie Scout — chat UI + thumbs feedback.

Talks only to the FastAPI backend (via frontend.client). No agent import.
Run: streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from frontend import client
from frontend.posters import poster_url

st.set_page_config(page_title="Movie Scout", page_icon="🎬")


def _render_citation(rec: dict) -> None:
    title = rec.get("title", "?")
    year = rec.get("year", "")
    with st.expander(f"{title} ({year})"):
        img = poster_url(rec["tmdb_id"]) if rec.get("tmdb_id") else None
        if img:
            st.image(img, width=180)
        st.markdown(rec.get("why_for_you", ""))
        hint = rec.get("provider_hint")
        if hint:
            st.caption(f"Where to watch: {hint}")


def _send_feedback(run_id: str, rating: str) -> None:
    try:
        client.feedback(run_id, rating)
    except Exception:  # noqa: BLE001 — feedback failure must not break the UI
        st.toast("Could not send feedback", icon="⚠️")
    st.session_state[f"fb_{run_id}"] = rating


def _render_entry(entry: dict) -> None:
    run_id = entry["run_id"]
    st.markdown(f"**You:** {entry['query']}")
    st.markdown(entry["answer"])

    for rec in entry.get("citations", []):
        _render_citation(rec)

    m = entry.get("metrics", {})
    st.caption(
        f"latency {m.get('latency_ms', 0):.0f} ms · "
        f"cost ${m.get('cost_usd', 0):.4f} · "
        f"{m.get('tool_calls', 0)} tool calls"
    )

    sent = st.session_state.get(f"fb_{run_id}")
    col_up, col_down = st.columns(2)
    col_up.button(
        "👍", key=f"up_{run_id}", disabled=sent is not None,
        on_click=_send_feedback, args=(run_id, "up"),
    )
    col_down.button(
        "👎", key=f"down_{run_id}", disabled=sent is not None,
        on_click=_send_feedback, args=(run_id, "down"),
    )
    if sent:
        st.caption(f"Feedback sent: {sent}")
    st.divider()


def _render_taste_sidebar() -> None:
    """Letterboxd export uploader — stored in session_state, never persisted."""
    with st.sidebar:
        st.header("Your Taste Profile")

        profile = st.session_state.get("taste_profile")
        if profile:
            st.success(
                f"Profile active: {profile.get('film_count', '?')} films"
            )
            if st.button("Clear profile (cold start)"):
                st.session_state.taste_profile = None
                st.rerun()
        else:
            st.caption(
                "Optional — specific queries (\"films like Inception\") work "
                "without it. Generic ones (\"recommend me a film\") need your "
                "taste profile to mean anything. Upload your Letterboxd export "
                "to personalise results toward films you've rated highly."
            )
            st.markdown("[Export your Letterboxd data](https://letterboxd.com/user/exportdata/)")

        uploaded = st.file_uploader(
            "Letterboxd ratings.csv or ZIP export",
            type=["csv", "zip"],
            help=(
                "Export your data at letterboxd.com/user/exportdata/. "
                "Upload ratings.csv or the full ZIP bundle. Optional."
            ),
        )

        if uploaded is not None and st.session_state.get("_taste_upload_id") != uploaded.file_id:
            st.session_state._taste_upload_id = uploaded.file_id
            with st.spinner("Building taste profile… (may take ~1 min)"):
                try:
                    response = client.upload_taste(
                        uploaded.getvalue(),
                        filename=uploaded.name,
                    )
                    st.session_state.taste_profile = response["profile"]
                    resolved = response.get("resolved", 0)
                    tmdb_miss = response.get("tmdb_miss", 0)
                    out_of_corpus = response.get("out_of_corpus", 0)
                    total = response.get("total_input", 0)
                    st.success(
                        f"Profile built: {resolved} films resolved, "
                        f"{tmdb_miss} title misses, "
                        f"{out_of_corpus} out-of-corpus "
                        f"(of {total} total)"
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Upload failed: {exc}")


def _render_clarify_turn(pending_query: str, question: str) -> None:
    """Render the franchise clarification question + follow-up input.

    On submit, calls client.ask with the pending query and the user's answer,
    then stores the resulting recommendation in history (AC-9).
    """
    st.info(question)
    with st.form("clarify_form", clear_on_submit=True):
        answer = st.text_input("Your answer…")
        submitted = st.form_submit_button("Submit")

    if submitted and answer.strip():
        sibling_ids: list[int] = st.session_state.get("franchise_sibling_ids", [])
        with st.spinner("Finding recommendations…"):
            try:
                resp = client.ask(
                    pending_query,
                    taste_profile=st.session_state.get("taste_profile"),
                    clarification_answer=answer.strip(),
                    franchise_sibling_ids=sibling_ids or None,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Backend error: {exc}")
                resp = None

        if resp is not None:
            # Clear the pending clarify state
            st.session_state.pop("pending_query", None)
            st.session_state.pop("franchise_sibling_ids", None)

            if resp.get("needs_clarification"):
                # Extremely unlikely (single-turn cap), but handle gracefully
                st.warning("Still waiting for clarification — please try answering again.")
            else:
                st.session_state.history.append(
                    {
                        "run_id": resp["run_id"],
                        "query": pending_query,
                        "answer": resp["final_answer"],
                        "citations": resp.get("citations", []),
                        "metrics": {
                            "latency_ms": resp.get("latency_ms", 0),
                            "cost_usd": resp.get("cost_usd", 0),
                            "tool_calls": resp.get("tool_calls", 0),
                        },
                    }
                )
                st.rerun()


def main() -> None:
    st.title("Movie Scout")

    _render_taste_sidebar()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Show taste status in main area when active
    profile = st.session_state.get("taste_profile")
    if profile:
        st.info(
            f"Taste-personalised search active "
            f"({profile.get('film_count', '?')} films). "
            f"Use the sidebar to clear or update."
        )

    # If a franchise clarify turn is pending, render the question form instead
    # of the normal ask form — the user must answer before getting recommendations.
    pending_query: str | None = st.session_state.get("pending_query")
    if pending_query is not None:
        pending_question: str = st.session_state.get(
            "pending_clarify_question", "Do you want franchise sequels included?"
        )
        _render_clarify_turn(pending_query, pending_question)
    else:
        with st.form("ask_form", clear_on_submit=True):
            query = st.text_input("Describe what you want to watch…")
            submitted = st.form_submit_button("Ask")

        if submitted and query.strip():
            with st.spinner("Thinking…"):
                try:
                    resp = client.ask(
                        query,
                        taste_profile=st.session_state.get("taste_profile"),
                    )
                except Exception as exc:  # noqa: BLE001 — surface, don't crash
                    st.error(f"Backend error: {exc}")
                    resp = None
            if resp is not None:
                if resp.get("needs_clarification"):
                    # Franchise clarify turn: stash pending state and rerun to
                    # show the clarification form instead of adding to history.
                    st.session_state.pending_query = query
                    st.session_state.pending_clarify_question = resp.get(
                        "clarification_question",
                        "Do you want franchise sequels included?",
                    )
                    st.session_state.franchise_sibling_ids = resp.get(
                        "franchise_sibling_ids", []
                    )
                    st.rerun()
                else:
                    st.session_state.history.append(
                        {
                            "run_id": resp["run_id"],
                            "query": query,
                            "answer": resp["final_answer"],
                            "citations": resp.get("citations", []),
                            "metrics": {
                                "latency_ms": resp.get("latency_ms", 0),
                                "cost_usd": resp.get("cost_usd", 0),
                                "tool_calls": resp.get("tool_calls", 0),
                            },
                        }
                    )

    # newest first
    for entry in reversed(st.session_state.history):
        _render_entry(entry)


main()
