from unittest.mock import patch, AsyncMock
import uuid

import pytest
from pydantic import TypeAdapter
from sqlalchemy import select

from src.db.models import Relationship as RelationshipModel
from src.db.schemas import Legislation
from src.rag.pipeline import _save_relationships, run_pipeline
from tests.conftest import VALID_LEGISLATION_JSON, VALID_RELATIONSHIPS_JSON


@pytest.fixture
def sample_legislation_with_relationships() -> Legislation:
    leg = Legislation.model_validate_json(VALID_LEGISLATION_JSON)
    from src.db.schemas import Relationship
    rels = TypeAdapter(list[Relationship]).validate_json(VALID_RELATIONSHIPS_JSON)
    leg.relationships = rels
    return leg


# ── run_pipeline ──────────────────────────────────────────────────────────────

class TestRunPipeline:

    @patch("src.rag.pipeline.add_legislation")
    @patch("src.rag.pipeline.run_ingestion_workflow")
    @patch("src.rag.pipeline.parse_pdf")
    async def test_returns_legislation_on_success(
        self, mock_parse, mock_ingest, mock_add, db_session, sample_legislation_with_relationships
    ):
        mock_parse.return_value = "some legislation text"
        mock_ingest.return_value = sample_legislation_with_relationships
        mock_add.return_value = None

        result = await run_pipeline("/fake/path.pdf", db_session)

        assert result is not None
        assert result.code == "LAW-88-2003"

    @patch("src.rag.pipeline.add_legislation")
    @patch("src.rag.pipeline.run_ingestion_workflow")
    @patch("src.rag.pipeline.parse_pdf")
    async def test_calls_each_step_in_order(
        self, mock_parse, mock_ingest, mock_add, db_session, sample_legislation_with_relationships
    ):
        mock_parse.return_value = "text"
        mock_ingest.return_value = sample_legislation_with_relationships
        mock_add.return_value = None

        await run_pipeline("/fake/path.pdf", db_session)

        mock_parse.assert_called_once_with("/fake/path.pdf")
        mock_ingest.assert_called_once_with("text")
        mock_add.assert_called_once_with(sample_legislation_with_relationships)

    @patch("src.rag.pipeline.run_ingestion_workflow")
    @patch("src.rag.pipeline.parse_pdf")
    async def test_returns_none_when_pdf_is_empty(
        self, mock_parse, mock_ingest, db_session
    ):
        mock_parse.return_value = ""

        result = await run_pipeline("/fake/path.pdf", db_session)

        assert result is None
        mock_ingest.assert_not_called()

    @patch("src.rag.pipeline.add_legislation")
    @patch("src.rag.pipeline.run_ingestion_workflow")
    @patch("src.rag.pipeline.parse_pdf")
    async def test_returns_none_when_ingestion_fails(
        self, mock_parse, mock_ingest, mock_add, db_session
    ):
        mock_parse.return_value = "some text"
        mock_ingest.return_value = None

        result = await run_pipeline("/fake/path.pdf", db_session)

        assert result is None
        mock_add.assert_not_called()

    @patch("src.rag.pipeline.add_legislation")
    @patch("src.rag.pipeline.run_ingestion_workflow")
    @patch("src.rag.pipeline.parse_pdf")
    async def test_saves_relationships_to_database(
        self, mock_parse, mock_ingest, mock_add, db_session, sample_legislation_with_relationships
    ):
        mock_parse.return_value = "text"
        mock_ingest.return_value = sample_legislation_with_relationships
        mock_add.return_value = None

        await run_pipeline("/fake/path.pdf", db_session)

        rows = (await db_session.execute(select(RelationshipModel))).scalars().all()
        assert len(rows) == len(sample_legislation_with_relationships.relationships)

    @patch("src.rag.pipeline.add_legislation")
    @patch("src.rag.pipeline.run_ingestion_workflow")
    @patch("src.rag.pipeline.parse_pdf")
    async def test_pipeline_with_no_relationships_saves_nothing(
        self, mock_parse, mock_ingest, mock_add, db_session
    ):
        legislation = Legislation.model_validate_json(VALID_LEGISLATION_JSON)
        legislation.relationships = []
        mock_parse.return_value = "text"
        mock_ingest.return_value = legislation
        mock_add.return_value = None

        await run_pipeline("/fake/path.pdf", db_session)

        rows = (await db_session.execute(select(RelationshipModel))).scalars().all()
        assert rows == []


# ── _save_relationships ───────────────────────────────────────────────────────

class TestSaveRelationships:

    async def test_saves_all_relationships(self, db_session):
        from src.db.schemas import Relationship
        rels = TypeAdapter(list[Relationship]).validate_json(VALID_RELATIONSHIPS_JSON)

        await _save_relationships(rels, db_session)

        rows = (await db_session.execute(select(RelationshipModel))).scalars().all()
        assert len(rows) == 1

    async def test_saves_correct_fields(self, db_session):
        from src.db.schemas import Relationship
        rels = TypeAdapter(list[Relationship]).validate_json(VALID_RELATIONSHIPS_JSON)

        await _save_relationships(rels, db_session)

        row = (await db_session.execute(select(RelationshipModel))).scalars().first()
        assert row.father_legislation == "LAW-12-2010"
        assert row.father_article == "3"
        assert row.affected_legislation == "LAW-88-2003"
        assert row.affected_article == "2"
        assert row.illustration == "applies when loan amount exceeds 100,000 units after 2010"

    async def test_saves_nothing_for_empty_list(self, db_session):
        await _save_relationships([], db_session)

        rows = (await db_session.execute(select(RelationshipModel))).scalars().all()
        assert rows == []

    async def test_each_relationship_gets_unique_id(self, db_session):
        from src.db.schemas import Relationship, RelationshipType
        rels = [
            Relationship(
                type=RelationshipType.AMENDS,
                father_legislation="LAW-A",
                affected_legislation="LAW-B",
                illustration="always applies",
            ),
            Relationship(
                type=RelationshipType.REFERENCES,
                father_legislation="LAW-C",
                affected_legislation="LAW-D",
                illustration="always applies",
            ),
        ]

        await _save_relationships(rels, db_session)

        rows = (await db_session.execute(select(RelationshipModel))).scalars().all()
        ids = [row.id for row in rows]
        assert len(set(ids)) == 2