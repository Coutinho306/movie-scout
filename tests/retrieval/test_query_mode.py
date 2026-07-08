"""Unit tests for agent.tools.query_mode.classify_query_mode.

All tests are deterministic (no network, no model calls). Representative
queries are drawn from each tier of the 120-query diagnostic suite.
"""

from __future__ import annotations

import pytest

from agent.tools.query_mode import classify_query_mode


# ---------------------------------------------------------------------------
# Tier 2 → hybrid (templated "a … film — …" shape)
# ---------------------------------------------------------------------------

class TestTier2Template:
    """Tier-2 queries always route to hybrid."""

    @pytest.mark.parametrize("query", [
        "a Mystery, Thriller, Drama film — You don't know what you've got 'til it's...",
        "a Adventure, Fantasy film — Dark secrets revealed.",
        "a Thriller, Mystery film — The holiest event of our time. Perfect for their return.",
        "a Crime, Thriller film — Revenge is a funny thing.",
        "a Horror, Thriller film — It is the greatest mystery of all.",
        "a Animation, Family, Adventure, Comedy film — For of such is the kingdom of heaven.",
        "an Action, Crime, Thriller film — Be careful who you trust.",
    ])
    def test_routes_to_hybrid(self, query: str) -> None:
        assert classify_query_mode(query) is True


# ---------------------------------------------------------------------------
# Tier 1 → hybrid (first overview sentence, narrative, > 8 tokens)
# ---------------------------------------------------------------------------

class TestTier1Overview:
    """Tier-1 overview sentences route to hybrid."""

    @pytest.mark.parametrize("query", [
        (
            "With his wife's disappearance having become the focus of an intense "
            "media circus, a man sees the spotlight shift when it's suspected that "
            "he may not be the grieving husband he appears to be."
        ),
        (
            "Dumbledore tries to prepare Harry for the final battle with Voldemort "
            "while Death Eaters wreak havoc in both the Muggle and wizarding worlds."
        ),
        (
            "Harvard symbologist Robert Langdon is recruited by the Vatican to "
            "investigate the apparent return of the Illuminati."
        ),
        (
            "Danny Ocean's team of criminals are back and composing a plan more "
            "personal than ever."
        ),
        (
            "Four unwitting heroes cross paths on their journey to the sleepy town "
            "of Silverado."
        ),
        (
            "Grappling with his past after a life of crime and murder, Roland "
            "confesses his story to a priest."
        ),
    ])
    def test_routes_to_hybrid(self, query: str) -> None:
        assert classify_query_mode(query) is True


# ---------------------------------------------------------------------------
# Tier 3 → dense (LLM-generated abstract / conversational)
# ---------------------------------------------------------------------------

class TestTier3Abstract:
    """Tier-3 abstract/conversational queries route to dense."""

    @pytest.mark.parametrize("query", [
        "I'm looking for a psychological thriller with dark twists and complex characters.",
        "I'm looking for a fantasy film with a darker tone that involves magic, friendship, and coming-of-age themes.",
        "I'm looking for a thrilling mystery film that involves secret societies, religious artifacts, and a race against time.",
        "I'm looking for a stylish heist movie with clever twists and a strong ensemble cast.",
        "I'm looking for a classic horror film that involves themes of supernatural evil and dark prophecy.",
        "Can you recommend a visually stunning documentary that showcases the beauty and diversity of the planet?",
        "Looking for a quirky romantic comedy with a blend of humor and drama set in a vibrant urban environment.",
        "I'm looking for a suspenseful thriller that explores themes of obsession and revenge with a dark, atmospheric tone.",
        "I'm looking for an action thriller that revolves around a man's quest for revenge after losing a loved one.",
        "I'm looking for a fun animated movie featuring quirky creatures and a lighthearted adventure.",
    ])
    def test_routes_to_dense(self, query: str) -> None:
        assert classify_query_mode(query) is False


# ---------------------------------------------------------------------------
# Tier 0 → dense (verbatim title, short)
# ---------------------------------------------------------------------------

class TestTier0Title:
    """Tier-0 verbatim title queries route to dense."""

    @pytest.mark.parametrize("query", [
        "Gone Girl",
        "Harry Potter and the Half-Blood Prince",
        "Angels & Demons",
        "Ocean's Thirteen",
        "The Omen",
        "Inception",
        "Avatar",
        "Titanic",
    ])
    def test_routes_to_dense(self, query: str) -> None:
        assert classify_query_mode(query) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case and boundary inputs."""

    def test_empty_string_returns_dense(self) -> None:
        assert classify_query_mode("") is False

    def test_whitespace_only_returns_dense(self) -> None:
        assert classify_query_mode("   ") is False

    def test_em_dash_variant_hybrid(self) -> None:
        # em-dash (—) variant of tier-2 template
        assert classify_query_mode("a Drama film — A story of loss.") is True

    def test_en_dash_variant_hybrid(self) -> None:
        # en-dash (–) variant
        assert classify_query_mode("a Comedy film – Finding the funny side.") is True

    def test_ascii_dash_variant_hybrid(self) -> None:
        # ASCII hyphen variant
        assert classify_query_mode("a Thriller film - No one is safe.") is True

    def test_find_me_prefix_dense(self) -> None:
        assert classify_query_mode("Find me a great sci-fi movie") is False

    def test_recommend_prefix_dense(self) -> None:
        assert classify_query_mode("Recommend something dark and atmospheric") is False

    def test_long_but_request_prefix_is_dense(self) -> None:
        # Even a long sentence starting with "I'm looking for" stays dense
        q = (
            "I'm looking for a dark psychological horror film that explores themes "
            "of possession and supernatural evil with a gripping atmosphere."
        )
        assert classify_query_mode(q) is False

    def test_short_descriptive_no_template_is_dense(self) -> None:
        # 8 tokens or fewer without tier-2 template → dense (title-like)
        assert classify_query_mode("slow meditative contemplative film") is False
