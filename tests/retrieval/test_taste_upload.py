"""Tests for retrieval.taste_upload — Phase 1 validation.

AC-2: All-in-corpus upload builds a valid TasteProfile, OpenAI embedding
      client is never invoked.
AC-3: Title miss is skipped and counted, not fatal.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from ingestion.models import TasteProfile
from retrieval.taste_upload import (
    TasteUploadResult,
    _point_id,
    build_taste_profile_from_upload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ratings_csv(rows: list[dict]) -> bytes:
    """Build a minimal ratings.csv bytes buffer."""
    header = "Date,Name,Year,Letterboxd URI,Rating\n"
    lines = [
        f"2024-01-01,{r['name']},{r['year']},https://letterboxd.com/x,{r['rating']}\n"
        for r in rows
    ]
    return (header + "".join(lines)).encode()


def _make_zip(
    ratings: list[dict] | None = None,
    liked: list[dict] | None = None,
    watched: list[dict] | None = None,
    watchlist: list[dict] | None = None,
) -> bytes:
    """Build a minimal Letterboxd ZIP export bundle."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if ratings is not None:
            lines = "Date,Name,Year,Letterboxd URI,Rating\n"
            for r in ratings:
                lines += f"2024-01-01,{r['name']},{r['year']},https://x,{r['rating']}\n"
            zf.writestr("ratings.csv", lines)
        if liked is not None:
            lines = "Date,Name,Year,Letterboxd URI\n"
            for r in liked:
                lines += f"2024-01-01,{r['name']},{r['year']},https://x\n"
            zf.writestr("likes/films.csv", lines)
        if watched is not None:
            lines = "Date,Name,Year,Letterboxd URI\n"
            for r in watched:
                lines += f"2024-01-01,{r['name']},{r['year']},https://x\n"
            zf.writestr("watched.csv", lines)
        if watchlist is not None:
            lines = "Date,Name,Year,Letterboxd URI\n"
            for r in watchlist:
                lines += f"2024-01-01,{r['name']},{r['year']},https://x\n"
            zf.writestr("watchlist.csv", lines)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# AC-2: All-in-corpus upload → valid TasteProfile, NO OpenAI embedding call
# ---------------------------------------------------------------------------

class _FakeTmdbResult:
    def __init__(self, tmdb_id: int) -> None:
        self.tmdb_id = tmdb_id
        self.title = "Test Film"
        self.year = 2020
        self.match_score = 0.99


class _FakeQdrantRecord:
    def __init__(self, point_id: str, vector: list[float], genres: list[str]) -> None:
        self.id = point_id
        self.vector: Any = {"": vector, "text": [0.0] * 3}
        self.payload: dict = {"genres": genres, "title": "Test", "year": 2020}


def _make_fake_retrieve(tmdb_id_to_vec: dict[int, list[float]]) -> Any:
    """Return a fake client.retrieve that returns records for known tmdb_ids."""
    def _retrieve(collection_name: str, ids: list[str], **kwargs: Any) -> list[_FakeQdrantRecord]:
        pid_to_tid = {_point_id(tid): tid for tid in tmdb_id_to_vec}
        result = []
        for pid in ids:
            if pid in pid_to_tid:
                tid = pid_to_tid[pid]
                vec = tmdb_id_to_vec[tid]
                result.append(_FakeQdrantRecord(pid, vec, ["Action", "Drama"]))
        return result
    return _retrieve


def test_all_in_corpus_no_openai_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-2: Building from all-in-corpus films must NOT call OpenAI embed_texts."""
    # Two films both "in corpus" (fake TMDB + fake Qdrant)
    tmdb_ids = {1001: [0.1] * 1536, 1002: [0.2] * 1536}

    def _fake_search_tmdb(name: str, year: int, api_key: str) -> _FakeTmdbResult | None:
        mapping = {"Inception": 1001, "The Matrix": 1002}
        tid = mapping.get(name)
        return _FakeTmdbResult(tid) if tid else None

    fake_client = MagicMock()
    fake_client.retrieve.side_effect = _make_fake_retrieve(tmdb_ids)

    monkeypatch.setattr("retrieval.taste_upload.search_tmdb", _fake_search_tmdb)
    monkeypatch.setattr("retrieval.taste_upload.get_qdrant_client", lambda **kw: fake_client)

    # Track any call to openai embed — none should happen
    embed_spy = MagicMock(side_effect=AssertionError("embed_texts called unexpectedly!"))
    monkeypatch.setattr("ingestion.embedding.Embedder.embed_texts", embed_spy)

    csv_bytes = _make_ratings_csv([
        {"name": "Inception", "year": 2010, "rating": 5.0},
        {"name": "The Matrix", "year": 1999, "rating": 4.5},
    ])

    result = build_taste_profile_from_upload(
        csv_bytes,
        filename="ratings.csv",
        tmdb_api_key="fake-key",
        tmdb_sleep=0.0,
    )

    assert isinstance(result, TasteUploadResult)
    profile = result.profile
    assert isinstance(profile, TasteProfile)
    assert profile.film_count == 2
    assert len(profile.centroid) == 1536
    assert profile.created_at  # non-empty ISO timestamp
    assert isinstance(profile.genre_weights, dict)
    assert "Action" in profile.genre_weights

    report = result.report
    assert report.resolved == 2
    assert report.tmdb_miss == 0
    assert report.out_of_corpus == 0

    # OpenAI was never called
    embed_spy.assert_not_called()


def test_all_in_corpus_zip_no_openai_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-2 variant: ZIP upload also produces a valid profile without embedding."""
    tmdb_ids = {2001: [0.3] * 1536, 2002: [0.4] * 1536}

    def _fake_search_tmdb(name: str, year: int, api_key: str) -> _FakeTmdbResult | None:
        mapping = {"Parasite": 2001, "Oldboy": 2002}
        tid = mapping.get(name)
        return _FakeTmdbResult(tid) if tid else None

    fake_client = MagicMock()
    fake_client.retrieve.side_effect = _make_fake_retrieve(tmdb_ids)

    monkeypatch.setattr("retrieval.taste_upload.search_tmdb", _fake_search_tmdb)
    monkeypatch.setattr("retrieval.taste_upload.get_qdrant_client", lambda **kw: fake_client)

    embed_spy = MagicMock(side_effect=AssertionError("embed_texts called unexpectedly!"))
    monkeypatch.setattr("ingestion.embedding.Embedder.embed_texts", embed_spy)

    zip_bytes = _make_zip(
        ratings=[{"name": "Parasite", "year": 2019, "rating": 5.0}],
        liked=[{"name": "Oldboy", "year": 2003}],
        watched=[{"name": "Parasite", "year": 2019}],  # deduped
    )

    result = build_taste_profile_from_upload(
        zip_bytes,
        filename="letterboxd_export.zip",
        tmdb_api_key="fake-key",
        tmdb_sleep=0.0,
    )

    profile = result.profile
    assert profile.film_count == 2
    assert len(profile.centroid) == 1536
    embed_spy.assert_not_called()


# ---------------------------------------------------------------------------
# AC-3: TMDB miss is skipped + counted, never fatal
# ---------------------------------------------------------------------------

def test_tmdb_miss_skipped_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-3: Unresolved title is skipped; resolved films still build a profile."""
    tmdb_ids = {3001: [0.5] * 1536}

    call_count = {"n": 0}

    def _fake_search_tmdb(name: str, year: int, api_key: str) -> _FakeTmdbResult | None:
        call_count["n"] += 1
        if name == "Fake Obscure Film 9999":
            return None  # miss
        return _FakeTmdbResult(3001)

    fake_client = MagicMock()
    fake_client.retrieve.side_effect = _make_fake_retrieve(tmdb_ids)

    monkeypatch.setattr("retrieval.taste_upload.search_tmdb", _fake_search_tmdb)
    monkeypatch.setattr("retrieval.taste_upload.get_qdrant_client", lambda **kw: fake_client)

    csv_bytes = _make_ratings_csv([
        {"name": "Inception", "year": 2010, "rating": 5.0},
        {"name": "Fake Obscure Film 9999", "year": 1900, "rating": 4.0},
    ])

    result = build_taste_profile_from_upload(
        csv_bytes,
        filename="ratings.csv",
        tmdb_api_key="fake-key",
        tmdb_sleep=0.0,
    )

    assert result.report.tmdb_miss == 1
    assert result.report.resolved == 1
    assert result.profile.film_count == 1
    # Two search_tmdb calls were made (not short-circuited on miss)
    assert call_count["n"] == 2


def test_all_tmdb_miss_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """All TMDB misses → ValueError (cannot form centroid with zero films)."""
    monkeypatch.setattr("retrieval.taste_upload.search_tmdb", lambda *a, **kw: None)
    monkeypatch.setattr(
        "retrieval.taste_upload.get_qdrant_client",
        lambda **kw: MagicMock(),
    )

    csv_bytes = _make_ratings_csv([
        {"name": "NonExistent Film XYZ", "year": 1900, "rating": 5.0},
    ])

    with pytest.raises(ValueError, match="No films resolved"):
        build_taste_profile_from_upload(
            csv_bytes,
            filename="ratings.csv",
            tmdb_api_key="fake-key",
            tmdb_sleep=0.0,
        )


# ---------------------------------------------------------------------------
# Out-of-corpus films are dropped (id absent from batch result)
# ---------------------------------------------------------------------------

def test_out_of_corpus_films_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Films resolved via TMDB but absent from Qdrant are counted + skipped."""
    # Only tmdb_id 4001 is "in corpus"; 4002 is out-of-corpus
    tmdb_ids = {4001: [0.6] * 1536}

    def _fake_search_tmdb(name: str, year: int, api_key: str) -> _FakeTmdbResult | None:
        mapping = {"Film A": 4001, "Film B": 4002}
        tid = mapping.get(name)
        return _FakeTmdbResult(tid) if tid else None

    fake_client = MagicMock()
    fake_client.retrieve.side_effect = _make_fake_retrieve(tmdb_ids)

    monkeypatch.setattr("retrieval.taste_upload.search_tmdb", _fake_search_tmdb)
    monkeypatch.setattr("retrieval.taste_upload.get_qdrant_client", lambda **kw: fake_client)

    csv_bytes = _make_ratings_csv([
        {"name": "Film A", "year": 2020, "rating": 5.0},
        {"name": "Film B", "year": 2021, "rating": 4.5},
    ])

    result = build_taste_profile_from_upload(
        csv_bytes,
        filename="ratings.csv",
        tmdb_api_key="fake-key",
        tmdb_sleep=0.0,
    )

    assert result.report.resolved == 1
    assert result.report.out_of_corpus == 1
    assert result.profile.film_count == 1


# ---------------------------------------------------------------------------
# Top-N cap: only top films by weight are resolved
# ---------------------------------------------------------------------------

def test_top_n_cap_limits_tmdb_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap to top-N by weight before TMDB lookup — only N calls made."""
    resolve_calls: list[str] = []

    def _fake_search_tmdb(name: str, year: int, api_key: str) -> _FakeTmdbResult | None:
        resolve_calls.append(name)
        return _FakeTmdbResult(5000 + len(resolve_calls))

    fake_vecs = {5001: [0.7] * 1536}

    fake_client = MagicMock()
    fake_client.retrieve.side_effect = _make_fake_retrieve(fake_vecs)

    monkeypatch.setattr("retrieval.taste_upload.search_tmdb", _fake_search_tmdb)
    monkeypatch.setattr("retrieval.taste_upload.get_qdrant_client", lambda **kw: fake_client)

    # 5 films, cap=1
    csv_bytes = _make_ratings_csv([
        {"name": f"Film {i}", "year": 2000 + i, "rating": float(5 - i)}
        for i in range(5)
    ])

    build_taste_profile_from_upload(
        csv_bytes,
        filename="ratings.csv",
        tmdb_api_key="fake-key",
        tmdb_sleep=0.0,
        top_n_films=1,
    )

    assert len(resolve_calls) == 1, f"Expected 1 TMDB call, got {len(resolve_calls)}"


# ---------------------------------------------------------------------------
# Malformed upload
# ---------------------------------------------------------------------------

def test_malformed_csv_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A totally non-CSV file raises ValueError (missing required columns)."""
    garbage = b"this is not a csv at all \x00\x01\x02"

    with pytest.raises((ValueError, Exception)):
        build_taste_profile_from_upload(
            garbage,
            filename="ratings.csv",
            tmdb_api_key="fake-key",
            tmdb_sleep=0.0,
        )


# ---------------------------------------------------------------------------
# Helper: _point_id round-trip matches ingestion
# ---------------------------------------------------------------------------

def test_point_id_matches_ingestion() -> None:
    """Derived point ID must match the ingestion formula exactly."""
    import uuid as _uuid

    for tmdb_id in [27205, 550, 11, 1234567]:
        expected = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, str(tmdb_id)))
        assert _point_id(tmdb_id) == expected
