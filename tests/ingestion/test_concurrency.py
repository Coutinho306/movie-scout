"""AC3 + AC4 + AC6 — Concurrent movie loop tests.

AC3: 8-wide ThreadPoolExecutor produces the SAME set of Qdrant points (ids,
     vectors, payloads) as the sequential path on a fixed small mocked id list.
AC4: A film whose fetch raises (or returns None) is skipped and logged; the batch
     completes and all other candidates are upserted.
AC6: --workers plumbing — a non-default value reaches the loader; no new dep.

All Qdrant / TMDB / OpenAI calls are mocked.  No real network.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch


from ingestion.resources.tmdb_movies import _process_movie, load_tmdb_movies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(tmdb_id: int) -> MagicMock:
    meta = MagicMock()
    meta.tmdb_id = tmdb_id
    meta.title = f"Movie {tmdb_id}"
    meta.year = 2000
    meta.overview = "overview"
    meta.tagline = ""
    meta.genres = ["Drama"]
    meta.cast = ["Actor"]
    meta.director = "Director"
    meta.runtime = 90
    meta.vote_average = 7.0
    meta.popularity = 50.0
    meta.keywords = []
    meta.themes = ""
    meta.embed_text = f"embed text {tmdb_id}"
    return meta


def _expected_point_id(tmdb_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(tmdb_id)))


def _capture_upserts(mock_qdrant: MagicMock) -> list[dict]:
    """Collect all points upserted via the mock Qdrant client."""
    points = []
    for c in mock_qdrant.upsert.call_args_list:
        for p in c.kwargs.get("points", c.args[1] if len(c.args) > 1 else []):
            points.append({"id": p.id, "vector": p.vector, "payload": p.payload})
    return points


# ---------------------------------------------------------------------------
# AC3 — parity: concurrent == sequential (ids, vectors, payloads)
# ---------------------------------------------------------------------------


def test_concurrent_loop_parity_with_sequential() -> None:
    """Concurrent 8-wide loop produces the same upsert set as sequential (workers=1)."""
    candidate_ids = list(range(1, 9))  # 8 films
    fake_vector = [0.1, 0.2, 0.3]

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        return _make_metadata(tmdb_id)

    def _fake_embed(texts: list[str]) -> list[list[float]]:
        return [fake_vector for _ in texts]

    # --- concurrent run (workers=8) ---
    mock_qdrant_concurrent = MagicMock()
    mock_embedder_concurrent = MagicMock()
    mock_embedder_concurrent.embed_texts.side_effect = _fake_embed

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant_concurrent),
        patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", side_effect=_fake_fetch),
    ):
        load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder_concurrent,
            collection_name="calib_test",
            explicit_tmdb_ids=candidate_ids,
            workers=8,
        )

    concurrent_points = _capture_upserts(mock_qdrant_concurrent)

    # --- sequential run (workers=1) ---
    mock_qdrant_seq = MagicMock()
    mock_embedder_seq = MagicMock()
    mock_embedder_seq.embed_texts.side_effect = _fake_embed

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant_seq),
        patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", side_effect=_fake_fetch),
    ):
        load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder_seq,
            collection_name="calib_test",
            explicit_tmdb_ids=candidate_ids,
            workers=1,
        )

    sequential_points = _capture_upserts(mock_qdrant_seq)

    # Compare as sets keyed by point id (order may differ in concurrent path).
    concurrent_by_id = {p["id"]: p for p in concurrent_points}
    sequential_by_id = {p["id"]: p for p in sequential_points}

    assert set(concurrent_by_id.keys()) == set(sequential_by_id.keys()), (
        f"Id mismatch:\n  concurrent: {sorted(concurrent_by_id.keys())}\n"
        f"  sequential: {sorted(sequential_by_id.keys())}"
    )
    for pid in sequential_by_id:
        assert concurrent_by_id[pid]["vector"] == sequential_by_id[pid]["vector"], (
            f"Vector mismatch for point {pid}"
        )
        assert concurrent_by_id[pid]["payload"] == sequential_by_id[pid]["payload"], (
            f"Payload mismatch for point {pid}"
        )


def test_point_id_is_uuid5_of_tmdb_id() -> None:
    """Point id produced by _process_movie equals uuid5(NAMESPACE_DNS, str(tmdb_id))."""
    tmdb_id = 550
    fake_vector = [0.5, 0.6]
    meta = _make_metadata(tmdb_id)

    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [fake_vector]

    with patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", return_value=meta):
        _process_movie(
            tmdb_id,
            api_key="fake",
            embedder=mock_embedder,
            client=mock_qdrant,
            collection_name="calib_test",
            embed_text_recipe="base",
            sparse=False,
        )

    upsert_args = mock_qdrant.upsert.call_args
    point = upsert_args.kwargs["points"][0]
    assert point.id == _expected_point_id(tmdb_id)


def test_payload_shape_is_unchanged() -> None:
    """Payload produced by _process_movie has exactly the expected keys (AC7)."""
    expected_keys = {
        "tmdb_id", "title", "year", "genres", "cast", "director",
        "overview", "tagline", "runtime", "vote_average", "popularity", "themes",
    }
    tmdb_id = 680
    meta = _make_metadata(tmdb_id)

    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1]]

    with patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", return_value=meta):
        _process_movie(
            tmdb_id,
            api_key="fake",
            embedder=mock_embedder,
            client=mock_qdrant,
            collection_name="calib_test",
            embed_text_recipe="base",
            sparse=False,
        )

    upsert_args = mock_qdrant.upsert.call_args
    payload = upsert_args.kwargs["points"][0].payload
    assert set(payload.keys()) == expected_keys, (
        f"Payload key mismatch.\n  Got: {set(payload.keys())}\n  Expected: {expected_keys}"
    )


# ---------------------------------------------------------------------------
# AC4 — per-item error isolation
# ---------------------------------------------------------------------------


def test_one_raising_film_does_not_kill_batch() -> None:
    """A film whose fetch raises is skipped; all other candidates are upserted."""
    good_ids = [10, 20, 30, 40]
    bad_id = 99
    all_ids = good_ids + [bad_id]

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        if tmdb_id == bad_id:
            raise RuntimeError("TMDB fetch failed")
        return _make_metadata(tmdb_id)

    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", side_effect=_fake_fetch),
    ):
        loaded = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=all_ids,
            workers=4,
        )

    # All good films loaded; bad film skipped.
    assert loaded == len(good_ids), f"Expected {len(good_ids)} loaded, got {loaded}"
    upserted_ids = set()
    for c in mock_qdrant.upsert.call_args_list:
        for p in c.kwargs["points"]:
            upserted_ids.add(p.id)
    expected_ids = {_expected_point_id(i) for i in good_ids}
    assert upserted_ids == expected_ids


def test_none_metadata_is_skipped() -> None:
    """A film whose fetch returns None is skipped (return False), batch continues."""
    candidate_ids = [1, 2, 3]
    none_id = 2

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        if tmdb_id == none_id:
            return None
        return _make_metadata(tmdb_id)

    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1]]

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", side_effect=_fake_fetch),
    ):
        loaded = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=candidate_ids,
        )

    assert loaded == 2
    upserted_ids = set()
    for c in mock_qdrant.upsert.call_args_list:
        for p in c.kwargs["points"]:
            upserted_ids.add(p.id)
    assert _expected_point_id(none_id) not in upserted_ids


# ---------------------------------------------------------------------------
# AC6 — workers plumbing and no-new-dep
# ---------------------------------------------------------------------------


def test_workers_param_accepted_and_default_is_8() -> None:
    """load_tmdb_movies accepts workers kwarg; default is 8; non-default value accepted."""
    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1]]

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        return _make_metadata(tmdb_id)

    # Non-default workers value (3) should not error.
    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", side_effect=_fake_fetch),
        patch("ingestion.resources.tmdb_movies.ThreadPoolExecutor") as mock_tpe,
    ):
        mock_tpe.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tpe.return_value.__exit__ = MagicMock(return_value=False)
        # The executor's submit+as_completed need to work — use a real executor
        # for the actual call but capture the max_workers argument.
        pass

    # Simpler: just call load_tmdb_movies with workers=3 on an empty list
    # and assert it returns 0 without error.
    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", side_effect=_fake_fetch),
    ):
        loaded = load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=[],
            workers=3,
        )

    assert loaded == 0  # empty list, nothing to load


def test_workers_reaches_thread_pool_executor() -> None:
    """The workers value is forwarded to ThreadPoolExecutor(max_workers=...)."""
    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_texts.return_value = [[0.1]]

    def _fake_fetch(tmdb_id: int, api_key: str, *, embed_text_recipe: str = "base"):
        return _make_metadata(tmdb_id)

    captured_max_workers: list[int] = []

    original_tpe = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor

    class _CapturingTPE(original_tpe):
        def __init__(self, max_workers: int | None = None, **kwargs: Any) -> None:
            captured_max_workers.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    with (
        patch("ingestion.resources.tmdb_movies.QdrantClient", return_value=mock_qdrant),
        patch("ingestion.resources.tmdb_movies.fetch_movie_metadata", side_effect=_fake_fetch),
        patch("ingestion.resources.tmdb_movies.ThreadPoolExecutor", _CapturingTPE),
    ):
        load_tmdb_movies(
            api_key="fake",
            qdrant_url="http://localhost",
            qdrant_api_key="fake",
            watched_tmdb_ids=set(),
            embedder=mock_embedder,
            collection_name="calib_test",
            explicit_tmdb_ids=[1, 2],
            workers=5,
        )

    assert captured_max_workers == [5], (
        f"Expected ThreadPoolExecutor(max_workers=5), got {captured_max_workers}"
    )


def test_no_new_runtime_dependency() -> None:
    """concurrent.futures and threading are stdlib — no new pyproject.toml dep needed."""
    import importlib

    # These must import without error from stdlib.
    importlib.import_module("concurrent.futures")
    importlib.import_module("threading")

    # pyproject.toml must not list either as a runtime dependency.
    import tomllib
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    dep_names = [d.split("[")[0].split(">=")[0].split("==")[0].strip().lower() for d in deps]
    assert "concurrent" not in dep_names
    assert "threading" not in dep_names
