from unittest.mock import patch, MagicMock

import pytest

from src.db.schemas import LegislationStatus, RelationshipType
from src.rag.embeddings import embed, embed_batch
from src.rag.graph import format_for_llm, get_children, get_parents
from src.rag.retriever import retrieve, retrieve_active, retrieve_hybrid, retrieve_exact, retrieve_active_hybrid
from src.rag.vectorstore import (
    ArticleResult,
    LegislationResult,
    SearchResult,
    add_legislation,
    bm25_search,
    delete_legislation,
    exact_search,
    get_article,
    get_legislation,
    hybrid_search,
    search,
)
from tests.conftest import make_relationship


# ── Embeddings ────────────────────────────────────────────────────────────────

class TestEmbeddings:

    def test_embed_returns_list_of_floats(self):
        result = embed("banking loan collateral requirements")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result)

    def test_embed_batch_returns_correct_count(self):
        texts = ["article one content", "article two content", "article three content"]

        result = embed_batch(texts)

        assert len(result) == 3
        assert all(isinstance(vec, list) for vec in result)
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_same_text_produces_same_vector(self):
        text = "loan interest rate regulation"

        v1 = embed(text)
        v2 = embed(text)

        assert v1 == v2

    def test_embed_different_texts_produce_different_vectors(self):
        v1 = embed("loan collateral requirements")
        v2 = embed("unrelated topic about agriculture")

        assert v1 != v2

    def test_embed_and_embed_batch_are_consistent(self):
        text = "bank lending regulations"

        single = embed(text)
        batch = embed_batch([text])

        assert single == batch[0]


# ── Vectorstore ───────────────────────────────────────────────────────────────

class TestVectorstore:

    def test_add_legislation_indexes_all_articles(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        assert test_collection.count() == len(sample_legislation.articles)

    def test_search_returns_relevant_results(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = search("loan collateral requirements", n_results=2)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.legislation_code == "LAW-88-2003" for r in results)

    def test_search_result_has_correct_metadata(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = search("real estate collateral loan", n_results=1)

        r = results[0]
        assert r.legislation_code == "LAW-88-2003"
        assert r.status == "active"
        assert r.subject == sample_legislation.subject
        assert r.article_number in sample_legislation.articles

    def test_search_with_status_filter_excludes_repealed(
        self, test_collection, sample_legislation, sample_legislation_repealed
    ):
        add_legislation(sample_legislation)
        add_legislation(sample_legislation_repealed)

        results = search("collateral requirements loan", n_results=10, filters={"status": "active"})

        codes = [r.legislation_code for r in results]
        assert "LAW-88-2003" in codes
        assert "LAW-01-1990" not in codes

    def test_search_with_status_filter_finds_repealed(
        self, test_collection, sample_legislation, sample_legislation_repealed
    ):
        add_legislation(sample_legislation)
        add_legislation(sample_legislation_repealed)

        results = search("loan threshold regulation", n_results=10, filters={"status": "repealed"})

        codes = [r.legislation_code for r in results]
        assert "LAW-01-1990" in codes
        assert "LAW-88-2003" not in codes

    def test_delete_legislation_removes_its_articles(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)
        assert test_collection.count() == len(sample_legislation.articles)

        delete_legislation(sample_legislation.code)

        assert test_collection.count() == 0

    def test_upsert_is_idempotent(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)
        add_legislation(sample_legislation)

        assert test_collection.count() == len(sample_legislation.articles)

    def test_add_legislation_with_no_articles_does_nothing(self, test_collection, sample_legislation):
        sample_legislation.articles = {}
        add_legislation(sample_legislation)

        assert test_collection.count() == 0


# ── Direct lookup ─────────────────────────────────────────────────────────────

class TestGetArticle:

    def test_returns_article_result_for_existing_article(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        result = get_article("LAW-88-2003", "2")

        assert isinstance(result, ArticleResult)
        assert result.legislation_code == "LAW-88-2003"
        assert result.article_number == "2"
        assert result.content == sample_legislation.articles["2"]

    def test_returns_correct_metadata(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        result = get_article("LAW-88-2003", "1")

        assert result.subject == sample_legislation.subject
        assert result.status == "active"
        assert result.issuer == "Central Bank"

    def test_returns_none_for_missing_article(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        result = get_article("LAW-88-2003", "99")

        assert result is None

    def test_returns_none_for_missing_legislation(self, test_collection):
        result = get_article("NON-EXISTENT", "1")

        assert result is None


class TestGetLegislation:

    def test_returns_legislation_result_for_existing_code(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        result = get_legislation("LAW-88-2003")

        assert isinstance(result, LegislationResult)
        assert result.code == "LAW-88-2003"

    def test_returns_all_articles(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        result = get_legislation("LAW-88-2003")

        assert set(result.articles.keys()) == set(sample_legislation.articles.keys())
        assert result.articles["1"] == sample_legislation.articles["1"]
        assert result.articles["2"] == sample_legislation.articles["2"]

    def test_returns_correct_metadata(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        result = get_legislation("LAW-88-2003")

        assert result.subject == sample_legislation.subject
        assert result.status == "active"
        assert result.issuer == "Central Bank"

    def test_returns_none_for_missing_legislation(self, test_collection):
        result = get_legislation("NON-EXISTENT")

        assert result is None


# ── Retriever ─────────────────────────────────────────────────────────────────

class TestRetriever:

    @patch("src.rag.retriever.search")
    def test_retrieve_calls_search_with_query(self, mock_search):
        mock_search.return_value = []

        retrieve("what are loan requirements", rewrite=False)

        mock_search.assert_called_once_with(
            "what are loan requirements", n_results=5, filters=None
        )

    @patch("src.rag.retriever.invoke")
    @patch("src.rag.retriever.search")
    def test_retrieve_with_rewrite_true_rewrites_query(self, mock_search, mock_invoke):
        mock_invoke.return_value = "loan collateral requirements legislation"
        mock_search.return_value = []

        retrieve("what do I need to get a loan?", rewrite=True)

        mock_invoke.assert_called_once()
        mock_search.assert_called_once_with(
            "loan collateral requirements legislation", n_results=5, filters=None
        )

    @patch("src.rag.retriever.invoke")
    @patch("src.rag.retriever.search")
    def test_retrieve_with_rewrite_false_skips_invoke(self, mock_search, mock_invoke):
        mock_search.return_value = []

        retrieve("loan requirements", rewrite=False)

        mock_invoke.assert_not_called()

    @patch("src.rag.retriever.search")
    def test_retrieve_passes_filters(self, mock_search):
        mock_search.return_value = []

        retrieve("loan rules", filters={"status": "active"}, rewrite=False)

        mock_search.assert_called_once_with(
            "loan rules", n_results=5, filters={"status": "active"}
        )

    @patch("src.rag.retriever.search")
    def test_retrieve_passes_n_results(self, mock_search):
        mock_search.return_value = []

        retrieve("loan rules", n_results=10, rewrite=False)

        mock_search.assert_called_once_with(
            "loan rules", n_results=10, filters=None
        )

    @patch("src.rag.retriever.search")
    def test_retrieve_active_applies_active_status_filter(self, mock_search):
        mock_search.return_value = []

        retrieve_active("loan collateral", rewrite=False)

        mock_search.assert_called_once_with(
            "loan collateral", n_results=5, filters={"status": "active"}
        )

    @patch("src.rag.retriever.search")
    def test_retrieve_returns_search_results(self, mock_search):
        expected = [
            SearchResult(
                article_id="LAW-88-2003_article_2",
                legislation_code="LAW-88-2003",
                article_number="2",
                content="All loans exceeding 50,000 units must be secured.",
                subject="Banking collateral",
                status="active",
                distance=0.12,
            )
        ]
        mock_search.return_value = expected

        result = retrieve("loan collateral", rewrite=False)

        assert result == expected


# ── Graph ─────────────────────────────────────────────────────────────────────

class TestGetParents:

    async def test_returns_direct_parent(self, db_session):
        rel = make_relationship("LAW-12-2010", "LAW-88-2003", illustration="always applies")
        db_session.add(rel)
        await db_session.commit()

        nodes = await get_parents(db_session, "LAW-88-2003")

        assert len(nodes) == 1
        assert nodes[0].legislation_code == "LAW-12-2010"
        assert nodes[0].depth == 1

    async def test_returns_empty_when_no_parents(self, db_session):
        nodes = await get_parents(db_session, "LAW-88-2003")

        assert nodes == []

    async def test_filters_by_article_number(self, db_session):
        rel_art2 = make_relationship("LAW-12-2010", "LAW-88-2003", affected_article="2")
        rel_art3 = make_relationship("LAW-05-2008", "LAW-88-2003", affected_article="3")
        db_session.add_all([rel_art2, rel_art3])
        await db_session.commit()

        nodes = await get_parents(db_session, "LAW-88-2003", article_number="2")

        assert len(nodes) == 1
        assert nodes[0].legislation_code == "LAW-12-2010"

    async def test_traverses_to_depth_2(self, db_session):
        rel1 = make_relationship("LAW-12-2010", "LAW-88-2003")
        rel2 = make_relationship("LAW-05-2008", "LAW-12-2010")
        db_session.add_all([rel1, rel2])
        await db_session.commit()

        nodes = await get_parents(db_session, "LAW-88-2003", k_depth=2)

        codes = [n.legislation_code for n in nodes]
        assert "LAW-12-2010" in codes
        assert "LAW-05-2008" in codes
        depth2_node = next(n for n in nodes if n.legislation_code == "LAW-05-2008")
        assert depth2_node.depth == 2

    async def test_does_not_exceed_k_depth(self, db_session):
        rel1 = make_relationship("LAW-12-2010", "LAW-88-2003")
        rel2 = make_relationship("LAW-05-2008", "LAW-12-2010")
        rel3 = make_relationship("LAW-01-1990", "LAW-05-2008")
        db_session.add_all([rel1, rel2, rel3])
        await db_session.commit()

        nodes = await get_parents(db_session, "LAW-88-2003", k_depth=2)

        codes = [n.legislation_code for n in nodes]
        assert "LAW-01-1990" not in codes

    async def test_prevents_cycles(self, db_session):
        rel1 = make_relationship("LAW-12-2010", "LAW-88-2003")
        rel2 = make_relationship("LAW-88-2003", "LAW-12-2010")  # circular
        db_session.add_all([rel1, rel2])
        await db_session.commit()

        nodes = await get_parents(db_session, "LAW-88-2003", k_depth=3)

        codes = [n.legislation_code for n in nodes]
        assert codes.count("LAW-12-2010") == 1
        assert "LAW-88-2003" not in codes

    async def test_includes_relationship_type_and_illustration(self, db_session):
        rel = make_relationship(
            "LAW-12-2010", "LAW-88-2003",
            rel_type=RelationshipType.REPEALS,
            illustration="applies after 2010 reforms",
        )
        db_session.add(rel)
        await db_session.commit()

        nodes = await get_parents(db_session, "LAW-88-2003")

        assert nodes[0].relationship_type == "repeals"
        assert nodes[0].illustration == "applies after 2010 reforms"


class TestGetChildren:

    async def test_returns_direct_child(self, db_session):
        rel = make_relationship("LAW-88-2003", "LAW-01-1990")
        db_session.add(rel)
        await db_session.commit()

        nodes = await get_children(db_session, "LAW-88-2003")

        assert len(nodes) == 1
        assert nodes[0].legislation_code == "LAW-01-1990"
        assert nodes[0].depth == 1

    async def test_returns_empty_when_no_children(self, db_session):
        nodes = await get_children(db_session, "LAW-88-2003")

        assert nodes == []

    async def test_filters_by_father_article(self, db_session):
        rel_art2 = make_relationship("LAW-88-2003", "LAW-01-1990", father_article="2")
        rel_art3 = make_relationship("LAW-88-2003", "LAW-05-2008", father_article="3")
        db_session.add_all([rel_art2, rel_art3])
        await db_session.commit()

        nodes = await get_children(db_session, "LAW-88-2003", article_number="2")

        assert len(nodes) == 1
        assert nodes[0].legislation_code == "LAW-01-1990"

    async def test_traverses_to_depth_2(self, db_session):
        rel1 = make_relationship("LAW-88-2003", "LAW-12-2010")
        rel2 = make_relationship("LAW-12-2010", "LAW-01-1990")
        db_session.add_all([rel1, rel2])
        await db_session.commit()

        nodes = await get_children(db_session, "LAW-88-2003", k_depth=2)

        codes = [n.legislation_code for n in nodes]
        assert "LAW-12-2010" in codes
        assert "LAW-01-1990" in codes

    async def test_prevents_cycles(self, db_session):
        rel1 = make_relationship("LAW-88-2003", "LAW-12-2010")
        rel2 = make_relationship("LAW-12-2010", "LAW-88-2003")  # circular
        db_session.add_all([rel1, rel2])
        await db_session.commit()

        nodes = await get_children(db_session, "LAW-88-2003", k_depth=3)

        codes = [n.legislation_code for n in nodes]
        assert codes.count("LAW-12-2010") == 1
        assert "LAW-88-2003" not in codes


# ── format_for_llm ────────────────────────────────────────────────────────────

class TestFormatForLlm:

    def test_empty_nodes_returns_no_related_message(self):
        result = format_for_llm([])

        assert result == "No related legislation found."

    def test_contains_legislation_code(self):
        from src.rag.graph import GraphNode
        nodes = [GraphNode("LAW-12-2010", None, "amends", "always applies", depth=1)]

        result = format_for_llm(nodes)

        assert "LAW-12-2010" in result

    def test_contains_relationship_type(self):
        from src.rag.graph import GraphNode
        nodes = [GraphNode("LAW-12-2010", "3", "repeals", "only after 2015", depth=1)]

        result = format_for_llm(nodes)

        assert "repeals" in result

    def test_contains_illustration(self):
        from src.rag.graph import GraphNode
        nodes = [GraphNode("LAW-12-2010", None, "amends", "only when loan exceeds 100k", depth=1)]

        result = format_for_llm(nodes)

        assert "only when loan exceeds 100k" in result

    def test_contains_article_number_when_present(self):
        from src.rag.graph import GraphNode
        nodes = [GraphNode("LAW-12-2010", "5", "amends", "always applies", depth=1)]

        result = format_for_llm(nodes)

        assert "Article 5" in result

    def test_groups_by_depth(self):
        from src.rag.graph import GraphNode
        nodes = [
            GraphNode("LAW-A", None, "amends", "always applies", depth=1),
            GraphNode("LAW-B", None, "references", "conditional", depth=2),
        ]

        result = format_for_llm(nodes)

        assert "Direct" in result
        assert "Depth 2" in result


# ── BM25 search ───────────────────────────────────────────────────────────────

class TestBM25Search:

    def test_returns_results_for_matching_terms(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = bm25_search("collateral loan", n_results=3)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_returns_empty_for_no_matching_terms(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = bm25_search("zxqvbnm irrelevant gibberish", n_results=3)

        assert results == []

    def test_result_has_correct_legislation_code(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = bm25_search("collateral requirements", n_results=2)

        assert all(r.legislation_code == "LAW-88-2003" for r in results)

    def test_result_score_is_positive(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = bm25_search("loan collateral", n_results=3)

        assert all(r.score > 0 for r in results)

    def test_results_ordered_by_score_descending(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = bm25_search("loan collateral real estate", n_results=3)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_filters_by_metadata(self, test_collection, sample_legislation, sample_legislation_repealed):
        add_legislation(sample_legislation)
        add_legislation(sample_legislation_repealed)

        # "secured" appears only in LAW-88-2003 article 2 → unambiguous BM25 signal
        results = bm25_search("secured banking", n_results=10, filters={"status": "active"})

        codes = [r.legislation_code for r in results]
        assert "LAW-88-2003" in codes
        assert "LAW-01-1990" not in codes

    def test_new_legislation_added_after_first_search_is_found(
        self, test_collection, sample_legislation, sample_legislation_repealed
    ):
        add_legislation(sample_legislation)
        bm25_search("loan", n_results=1)  # triggers lazy load

        add_legislation(sample_legislation_repealed)

        results = bm25_search("repealed regulation threshold", n_results=5)
        codes = [r.legislation_code for r in results]
        assert "LAW-01-1990" in codes


# ── Exact search ──────────────────────────────────────────────────────────────

class TestExactSearch:

    def test_returns_articles_containing_exact_keyword(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = exact_search("real estate", n_results=5)

        assert len(results) > 0
        assert all("real estate" in r.content for r in results)

    def test_returns_empty_when_keyword_not_present(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = exact_search("nonexistent_keyword_xyz", n_results=5)

        assert results == []

    def test_match_is_case_sensitive(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        lower = exact_search("real estate", n_results=5)
        upper = exact_search("REAL ESTATE", n_results=5)

        assert len(lower) > 0
        assert len(upper) == 0

    def test_filters_exclude_repealed(
        self, test_collection, sample_legislation, sample_legislation_repealed
    ):
        add_legislation(sample_legislation)
        add_legislation(sample_legislation_repealed)

        results = exact_search("loan", n_results=10, filters={"status": "active"})

        codes = [r.legislation_code for r in results]
        assert "LAW-88-2003" in codes
        assert "LAW-01-1990" not in codes

    def test_result_has_score_of_one(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = exact_search("loans", n_results=5)

        assert all(r.score == 1.0 for r in results)

    def test_respects_n_results_limit(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = exact_search("loan", n_results=1)

        assert len(results) <= 1


# ── Hybrid search ─────────────────────────────────────────────────────────────

class TestHybridSearch:

    def test_returns_search_results(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = hybrid_search("loan collateral requirements", n_results=3)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_have_rrf_score(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = hybrid_search("loan collateral", n_results=3)

        assert all(r.score > 0 for r in results)

    def test_respects_n_results(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = hybrid_search("loan", n_results=2)

        assert len(results) <= 2

    def test_with_keyword_boosts_exact_match(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        with_keyword = hybrid_search("loan", keyword="real estate", n_results=3)
        article_ids = [r.article_id for r in with_keyword]

        assert any("article_2" in aid for aid in article_ids)

    def test_applies_filters(
        self, test_collection, sample_legislation, sample_legislation_repealed
    ):
        add_legislation(sample_legislation)
        add_legislation(sample_legislation_repealed)

        results = hybrid_search("loan collateral", n_results=10, filters={"status": "active"})

        codes = [r.legislation_code for r in results]
        assert "LAW-88-2003" in codes
        assert "LAW-01-1990" not in codes

    def test_no_duplicate_articles_in_results(self, test_collection, sample_legislation):
        add_legislation(sample_legislation)

        results = hybrid_search("loan collateral requirements", n_results=5)

        ids = [r.article_id for r in results]
        assert len(ids) == len(set(ids))

    def test_returns_empty_for_empty_collection(self, test_collection):
        results = hybrid_search("loan requirements", n_results=3)

        assert results == []


# ── Retriever hybrid/exact ─────────────────────────────────────────────────────

class TestRetrieverHybrid:

    @patch("src.rag.retriever.hybrid_search")
    def test_retrieve_hybrid_calls_hybrid_search(self, mock_hybrid):
        mock_hybrid.return_value = []

        retrieve_hybrid("loan requirements", rewrite=False)

        mock_hybrid.assert_called_once_with(
            "loan requirements", keyword=None, n_results=5, filters=None
        )

    @patch("src.rag.retriever.hybrid_search")
    def test_retrieve_hybrid_passes_keyword(self, mock_hybrid):
        mock_hybrid.return_value = []

        retrieve_hybrid("loan requirements", keyword="collateral", rewrite=False)

        mock_hybrid.assert_called_once_with(
            "loan requirements", keyword="collateral", n_results=5, filters=None
        )

    @patch("src.rag.retriever.hybrid_search")
    def test_retrieve_active_hybrid_passes_active_filter(self, mock_hybrid):
        mock_hybrid.return_value = []

        retrieve_active_hybrid("loan requirements", rewrite=False)

        mock_hybrid.assert_called_once_with(
            "loan requirements", keyword=None, n_results=5, filters={"status": "active"}
        )

    @patch("src.rag.retriever.invoke")
    @patch("src.rag.retriever.hybrid_search")
    def test_retrieve_hybrid_rewrites_query(self, mock_hybrid, mock_invoke):
        mock_invoke.return_value = "collateral loan legislation"
        mock_hybrid.return_value = []

        retrieve_hybrid("what about loans?", rewrite=True)

        mock_invoke.assert_called_once()
        mock_hybrid.assert_called_once_with(
            "collateral loan legislation", keyword=None, n_results=5, filters=None
        )

    @patch("src.rag.retriever.exact_search")
    def test_retrieve_exact_applies_active_filter(self, mock_exact):
        mock_exact.return_value = []

        retrieve_exact("real estate")

        mock_exact.assert_called_once_with(
            "real estate", n_results=10, filters={"status": "active"}
        )