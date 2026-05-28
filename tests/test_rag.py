from unittest.mock import patch, MagicMock

import pytest

from src.db.schemas import LegislationStatus, RelationshipType
from src.rag.embeddings import embed, embed_batch
from src.rag.graph import format_for_llm, get_children, get_parents
from src.rag.retriever import retrieve, retrieve_active
from src.rag.vectorstore import SearchResult, add_legislation, delete_legislation, search
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