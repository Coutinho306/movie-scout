# Movie Scout — Streamlit frontend

Chat-style UI over the FastAPI backend. Type what you want to watch, get a
recommendation with citation cards and thumbs up/down feedback.

## Run

```bash
streamlit run frontend/streamlit_app.py
```

Opens on http://localhost:8501. The FastAPI backend must be running (see the
repo README / spec 0007):

```bash
uvicorn api.fastapi_app:app --reload
```

## Environment

| Var            | Default                 | Purpose                                   |
|----------------|-------------------------|-------------------------------------------|
| `API_BASE_URL` | `http://localhost:8000` | Backend base URL for `/ask` + `/feedback` |
| `TMDB_API_KEY` | _(unset)_               | Enables poster images; omit to skip them  |

Posters are opportunistic — with no `TMDB_API_KEY`, cards render without images.

## Layering

The frontend talks to the API only. It does **not** import `agent`, LangChain,
or Qdrant — all HTTP goes through `frontend/client.py`.
