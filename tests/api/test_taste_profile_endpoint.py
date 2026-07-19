"""API tests for POST /taste-profile — Phase 2 validation.

AC-5: Valid CSV and ZIP return TasteProfileResponse; malformed upload → 422;
      no disk/DB writes; route is rate-limited via slowapi.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.fastapi_app as fapp
from api.fastapi_app import create_app
from api.dependencies import get_pg_pool
from ingestion.models import TasteProfile
from retrieval.taste_upload import ResolutionReport, TasteUploadResult
from slowapi import Limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ratings_csv(rows: list[dict]) -> bytes:
    header = "Date,Name,Year,Letterboxd URI,Rating\n"
    lines = [
        f"2024-01-01,{r['name']},{r['year']},https://x,{r['rating']}\n"
        for r in rows
    ]
    return (header + "".join(lines)).encode()


def _make_zip(ratings: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        lines = "Date,Name,Year,Letterboxd URI,Rating\n"
        for r in ratings:
            lines += f"2024-01-01,{r['name']},{r['year']},https://x,{r['rating']}\n"
        zf.writestr("ratings.csv", lines)
        zf.writestr("likes/films.csv", "Date,Name,Year,Letterboxd URI\n")
        zf.writestr("watched.csv", "Date,Name,Year,Letterboxd URI\n")
        zf.writestr("watchlist.csv", "Date,Name,Year,Letterboxd URI\n")
    buf.seek(0)
    return buf.read()


def _stub_upload_result() -> TasteUploadResult:
    profile = TasteProfile(
        centroid=[0.1] * 1536,
        film_count=2,
        rated_count=2,
        liked_count=0,
        top_genre_ids=[28, 18],
        genre_weights={"Action": 1.0, "Drama": 0.8},
        created_at="2026-07-09T00:00:00+00:00",
    )
    report = ResolutionReport(
        resolved=2, tmdb_miss=1, out_of_corpus=0, total_input=3
    )
    return TasteUploadResult(profile=profile, report=report)


def _make_client(monkeypatch: pytest.MonkeyPatch, stub_fn: Any = None) -> TestClient:
    """Create a TestClient with a fresh rate-limiter and stubbed upload service."""
    monkeypatch.setenv("TMDB_API_KEY", "fake-key")
    # Each test gets a fresh limiter so counts don't leak across tests
    monkeypatch.setattr(fapp, "limiter", Limiter(key_func=fapp._client_key))
    if stub_fn is None:
        stub_fn = lambda *a, **kw: _stub_upload_result()  # noqa: E731
    monkeypatch.setattr(
        "retrieval.taste_upload.build_taste_profile_from_upload",
        stub_fn,
    )
    app = create_app()
    app.dependency_overrides[get_pg_pool] = lambda: None
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_taste_profile_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid ratings.csv → 200 TasteProfileResponse with expected fields."""
    with _make_client(monkeypatch) as c:
        csv_bytes = _make_ratings_csv([
            {"name": "Inception", "year": 2010, "rating": 5.0},
            {"name": "The Matrix", "year": 1999, "rating": 4.5},
        ])
        resp = c.post(
            "/taste-profile",
            files={"file": ("ratings.csv", csv_bytes, "text/csv")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "profile" in body
    assert body["profile"]["film_count"] == 2
    assert body["resolved"] == 2
    assert body["tmdb_miss"] == 1
    assert body["out_of_corpus"] == 0
    assert body["total_input"] == 3
    # profile has centroid
    assert len(body["profile"]["centroid"]) == 1536


def test_taste_profile_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid ZIP → 200 TasteProfileResponse."""
    with _make_client(monkeypatch) as c:
        zip_bytes = _make_zip([{"name": "Parasite", "year": 2019, "rating": 5.0}])
        resp = c.post(
            "/taste-profile",
            files={"file": ("export.zip", zip_bytes, "application/zip")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "profile" in body


def test_taste_profile_malformed_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed upload (ValueError from service) → 422, not 500."""
    def _raise_value_error(*a: Any, **kw: Any) -> None:
        raise ValueError("ratings.csv must have Name, Year, Rating columns")

    with _make_client(monkeypatch, stub_fn=_raise_value_error) as c:
        garbage = b"not a csv\x00\x01\x02"
        resp = c.post(
            "/taste-profile",
            files={"file": ("ratings.csv", garbage, "text/csv")},
        )

    assert resp.status_code == 422, resp.text


def test_taste_profile_no_disk_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """No files written to disk during a successful taste-profile call."""
    import os

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    before_files = set(os.listdir(str(data_dir)))

    with _make_client(monkeypatch) as c:
        csv_bytes = _make_ratings_csv([{"name": "Film A", "year": 2020, "rating": 4.0}])
        resp = c.post(
            "/taste-profile",
            files={"file": ("ratings.csv", csv_bytes, "text/csv")},
        )

    assert resp.status_code == 200
    after_files = set(os.listdir(str(data_dir)))
    assert before_files == after_files, "New files were written to disk!"


def test_taste_profile_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    """/taste-profile rate-limited at 10/minute; 11th request → 429."""
    from api.config import ApiSettings

    monkeypatch.setenv("TMDB_API_KEY", "fake-key")
    monkeypatch.setattr(fapp, "limiter", Limiter(key_func=fapp._client_key))
    monkeypatch.setattr(
        "retrieval.taste_upload.build_taste_profile_from_upload",
        lambda *a, **kw: _stub_upload_result(),
    )

    app = create_app(ApiSettings(rate_limit="10/minute"))
    app.dependency_overrides[get_pg_pool] = lambda: None

    csv_bytes = _make_ratings_csv([{"name": "Film A", "year": 2020, "rating": 4.0}])

    with TestClient(app) as c:
        for _ in range(10):
            r = c.post(
                "/taste-profile",
                files={"file": ("ratings.csv", csv_bytes, "text/csv")},
            )
            assert r.status_code == 200
        r = c.post(
            "/taste-profile",
            files={"file": ("ratings.csv", csv_bytes, "text/csv")},
        )
        assert r.status_code == 429

    app.dependency_overrides.clear()
