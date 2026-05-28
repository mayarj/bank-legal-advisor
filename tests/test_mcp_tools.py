import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# FastMCP has a Pydantic/Python-3.13 compatibility bug that raises TypeError at
# import time. Stub it before the src.mcp chain is imported so the tool functions
# (which are plain Python callables) remain testable without a running MCP server.
_mock_mcp = MagicMock()
_mock_mcp.tool.return_value = lambda f: f  # @mcp.tool() becomes identity decorator
sys.modules.setdefault("fastmcp", MagicMock(FastMCP=MagicMock(return_value=_mock_mcp)))

from src.mcp.tools.legal_lookup import (
    _format_article,
    _format_legislation,
    _format_search_results,
    get_article_data,
    get_legislation_data,
    get_relationship_map,
    similarity_search,
)
from src.rag.vectorstore import ArticleResult, LegislationResult, SearchResult


# ── Formatter: _format_article ────────────────────────────────────────────────

class TestFormatArticle:

    def _sample(self) -> ArticleResult:
        return ArticleResult(
            legislation_code="LAW-88-2003",
            article_number="2",
            content="All loans exceeding 50,000 units must be secured.",
            subject="Banking collateral requirements",
            status="active",
            issuer="Central Bank",
            date="2003-06-15",
        )

    def test_contains_legislation_code(self):
        assert "LAW-88-2003" in _format_article(self._sample())

    def test_contains_article_number_and_content(self):
        result = _format_article(self._sample())
        assert "Article 2" in result
        assert "All loans exceeding 50,000 units must be secured." in result

    def test_contains_all_metadata_fields(self):
        result = _format_article(self._sample())
        assert "Banking collateral requirements" in result
        assert "active" in result
        assert "Central Bank" in result
        assert "2003-06-15" in result


# ── Formatter: _format_legislation ───────────────────────────────────────────

class TestFormatLegislation:

    def test_contains_legislation_code_and_metadata(self):
        data = LegislationResult(
            code="LAW-88-2003", subject="Collateral", status="active",
            issuer="Central Bank", date="2003-06-15", articles={}
        )
        result = _format_legislation(data)
        assert "LAW-88-2003" in result
        assert "Collateral" in result
        assert "Central Bank" in result

    def test_contains_all_articles(self):
        data = LegislationResult(
            code="LAW-88-2003", subject="Collateral", status="active",
            issuer="Central Bank", date="2003-06-15",
            articles={"1": "First article text.", "2": "Second article text."},
        )
        result = _format_legislation(data)
        assert "Article 1" in result
        assert "First article text." in result
        assert "Article 2" in result
        assert "Second article text." in result


# ── Formatter: _format_search_results ────────────────────────────────────────

class TestFormatSearchResults:

    def _sample(self) -> list[SearchResult]:
        return [
            SearchResult(
                article_id="LAW-88-2003_article_2",
                legislation_code="LAW-88-2003",
                article_number="2",
                content="Loans must be secured by real estate.",
                subject="Collateral rules",
                status="active",
                distance=0.1,
            )
        ]

    def test_produces_format_parseable_by_agent(self):
        # The agent's _parse_article_refs reads [CODE | Article N | STATUS]
        result = _format_search_results(self._sample())
        assert "[LAW-88-2003 | Article 2 | active]" in result

    def test_contains_article_content(self):
        result = _format_search_results(self._sample())
        assert "Loans must be secured by real estate." in result

    def test_multiple_results_are_separated(self):
        results = [
            SearchResult("id1", "LAW-A", "1", "Content A", "Subject A", "active", 0.1),
            SearchResult("id2", "LAW-B", "3", "Content B", "Subject B", "active", 0.2),
        ]
        result = _format_search_results(results)
        assert "LAW-A" in result
        assert "LAW-B" in result
        assert "---" in result


# ── get_article_data ──────────────────────────────────────────────────────────

class TestGetArticleData:

    @patch("src.mcp.tools.legal_lookup.get_article")
    def test_returns_formatted_string_for_existing_article(self, mock_get):
        mock_get.return_value = ArticleResult(
            legislation_code="LAW-88-2003", article_number="2",
            content="All loans exceeding 50k.", subject="Collateral",
            status="active", issuer="Central Bank", date="2003-06-15",
        )
        result = get_article_data("LAW-88-2003", "2")
        assert "LAW-88-2003" in result
        assert "Article 2" in result
        assert "All loans exceeding 50k." in result

    @patch("src.mcp.tools.legal_lookup.get_article")
    def test_returns_not_found_message_when_article_missing(self, mock_get):
        mock_get.return_value = None
        result = get_article_data("LAW-88-2003", "99")
        assert "not found" in result.lower()
        assert "99" in result

    @patch("src.mcp.tools.legal_lookup.get_article")
    def test_passes_correct_args_to_vectorstore(self, mock_get):
        mock_get.return_value = None
        get_article_data("LAW-88-2003", "5")
        mock_get.assert_called_once_with("LAW-88-2003", "5")


# ── get_legislation_data ──────────────────────────────────────────────────────

class TestGetLegislationData:

    @patch("src.mcp.tools.legal_lookup.get_legislation")
    def test_returns_formatted_string_for_existing_legislation(self, mock_get):
        mock_get.return_value = LegislationResult(
            code="LAW-88-2003", subject="Collateral", status="active",
            issuer="Central Bank", date="2003-06-15",
            articles={"1": "Article text."},
        )
        result = get_legislation_data("LAW-88-2003")
        assert "LAW-88-2003" in result

    @patch("src.mcp.tools.legal_lookup.get_legislation")
    def test_returns_not_found_message_when_legislation_missing(self, mock_get):
        mock_get.return_value = None
        result = get_legislation_data("NON-EXISTENT")
        assert "not found" in result.lower()
        assert "NON-EXISTENT" in result


# ── get_relationship_map ──────────────────────────────────────────────────────

class TestGetRelationshipMap:

    @patch("src.mcp.tools.legal_lookup.get_parents")
    @patch("src.mcp.tools.legal_lookup.format_for_llm")
    @patch("src.mcp.tools.legal_lookup.AsyncSessionFactory")
    async def test_calls_get_parents_when_parents_true(
        self, mock_factory, mock_format, mock_parents
    ):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_parents.return_value = []
        mock_format.return_value = "No related legislation found."

        await get_relationship_map("LAW-88-2003", article_number="2", k_depth=2, parents=True)

        mock_parents.assert_called_once_with(mock_session, "LAW-88-2003", "2", 2)

    @patch("src.mcp.tools.legal_lookup.get_children")
    @patch("src.mcp.tools.legal_lookup.format_for_llm")
    @patch("src.mcp.tools.legal_lookup.AsyncSessionFactory")
    async def test_calls_get_children_when_parents_false(
        self, mock_factory, mock_format, mock_children
    ):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_children.return_value = []
        mock_format.return_value = "No related legislation found."

        await get_relationship_map("LAW-88-2003", article_number="2", k_depth=2, parents=False)

        mock_children.assert_called_once_with(mock_session, "LAW-88-2003", "2", 2)

    @patch("src.mcp.tools.legal_lookup.get_parents")
    @patch("src.mcp.tools.legal_lookup.format_for_llm")
    @patch("src.mcp.tools.legal_lookup.AsyncSessionFactory")
    async def test_returns_formatted_string(
        self, mock_factory, mock_format, mock_parents
    ):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_parents.return_value = []
        mock_format.return_value = "Depth 1 — LAW-12-2010 (amends) always applies"

        result = await get_relationship_map("LAW-88-2003", parents=True)

        assert result == "Depth 1 — LAW-12-2010 (amends) always applies"


# ── similarity_search ─────────────────────────────────────────────────────────

class TestSimilaritySearch:

    @patch("src.mcp.tools.legal_lookup.retrieve_active")
    def test_returns_formatted_results(self, mock_retrieve):
        mock_retrieve.return_value = [
            SearchResult("id1", "LAW-88-2003", "2", "Loan content", "Subject", "active", 0.1)
        ]
        result = similarity_search("loan collateral")
        assert "[LAW-88-2003 | Article 2 | active]" in result

    @patch("src.mcp.tools.legal_lookup.retrieve_active")
    def test_returns_no_results_message_when_empty(self, mock_retrieve):
        mock_retrieve.return_value = []
        result = similarity_search("completely unrelated query")
        assert "No relevant legislation found" in result

    @patch("src.mcp.tools.legal_lookup.retrieve_active")
    def test_passes_all_params_to_retrieve_active(self, mock_retrieve):
        mock_retrieve.return_value = []
        similarity_search("test query", n_results=10, rewrite=False)
        mock_retrieve.assert_called_once_with("test query", n_results=10, rewrite=False)