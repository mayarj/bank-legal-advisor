from unittest.mock import patch

import pytest

from src.db.schemas import Legislation, Relationship
from src.rag.ingestion import (
    generate_legislation_data,
    generate_relationships_data,
    run_ingestion_workflow,
)
from tests.conftest import INVALID_JSON, VALID_LEGISLATION_JSON, VALID_RELATIONSHIPS_JSON


# ── generate_legislation_data ─────────────────────────────────────────────────

class TestGenerateLegislationData:

    @patch("src.rag.ingestion.invoke")
    def test_returns_legislation_object_on_valid_json(self, mock_invoke):
        mock_invoke.return_value = VALID_LEGISLATION_JSON

        result = generate_legislation_data("some legislation text")

        assert isinstance(result, Legislation)
        assert result.code == "LAW-88-2003"
        assert result.issuer == "Central Bank"
        assert len(result.articles) == 3

    @patch("src.rag.ingestion.invoke")
    def test_returns_correct_articles_dict(self, mock_invoke):
        mock_invoke.return_value = VALID_LEGISLATION_JSON

        result = generate_legislation_data("some text")

        assert "1" in result.articles
        assert "2" in result.articles
        assert "loan" in result.articles["2"].lower()

    @patch("src.rag.ingestion.invoke")
    def test_triggers_repair_on_first_bad_json_then_succeeds(self, mock_invoke):
        mock_invoke.side_effect = [INVALID_JSON, VALID_LEGISLATION_JSON]

        result = generate_legislation_data("some text")

        assert result is not None
        assert result.code == "LAW-88-2003"
        assert mock_invoke.call_count == 2

    @patch("src.rag.ingestion.invoke")
    def test_returns_none_after_max_retries_all_fail(self, mock_invoke):
        mock_invoke.return_value = INVALID_JSON

        result = generate_legislation_data("some text")

        assert result is None
        assert mock_invoke.call_count == 3  # 1 extraction + 2 repair attempts


# ── generate_relationships_data ───────────────────────────────────────────────

class TestGenerateRelationshipsData:

    @patch("src.rag.ingestion.invoke")
    def test_returns_list_of_relationship_objects(self, mock_invoke):
        mock_invoke.return_value = VALID_RELATIONSHIPS_JSON

        result = generate_relationships_data("some text")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Relationship)

    @patch("src.rag.ingestion.invoke")
    def test_returns_correct_relationship_fields(self, mock_invoke):
        mock_invoke.return_value = VALID_RELATIONSHIPS_JSON

        result = generate_relationships_data("some text")

        rel = result[0]
        assert rel.type.value == "amends"
        assert rel.father_legislation == "LAW-12-2010"
        assert rel.father_article == "3"
        assert rel.affected_legislation == "LAW-88-2003"
        assert rel.affected_article == "2"

    @patch("src.rag.ingestion.invoke")
    def test_returns_empty_list_for_no_relationships(self, mock_invoke):
        mock_invoke.return_value = "[]"

        result = generate_relationships_data("some text")

        assert result == []

    @patch("src.rag.ingestion.invoke")
    def test_returns_empty_list_after_max_retries(self, mock_invoke):
        mock_invoke.return_value = INVALID_JSON

        result = generate_relationships_data("some text")

        assert result == []
        assert mock_invoke.call_count == 3

    @patch("src.rag.ingestion.invoke")
    def test_triggers_repair_then_succeeds(self, mock_invoke):
        mock_invoke.side_effect = [INVALID_JSON, VALID_RELATIONSHIPS_JSON]

        result = generate_relationships_data("some text")

        assert len(result) == 1
        assert mock_invoke.call_count == 2


# ── run_ingestion_workflow ────────────────────────────────────────────────────

class TestRunIngestionWorkflow:

    @patch("src.rag.ingestion.generate_relationships_data")
    @patch("src.rag.ingestion.generate_legislation_data")
    def test_attaches_relationships_to_legislation(self, mock_leg, mock_rel):
        legislation = Legislation.model_validate_json(VALID_LEGISLATION_JSON)
        from pydantic import TypeAdapter
        relationships = TypeAdapter(list[Relationship]).validate_json(VALID_RELATIONSHIPS_JSON)

        mock_leg.return_value = legislation
        mock_rel.return_value = relationships

        result = run_ingestion_workflow("some text")

        assert result is not None
        assert len(result.relationships) == 1
        assert result.relationships[0].father_legislation == "LAW-12-2010"

    @patch("src.rag.ingestion.generate_relationships_data")
    @patch("src.rag.ingestion.generate_legislation_data")
    def test_returns_none_when_legislation_extraction_fails(self, mock_leg, mock_rel):
        mock_leg.return_value = None
        mock_rel.return_value = []

        result = run_ingestion_workflow("some text")

        assert result is None

    @patch("src.rag.ingestion.generate_relationships_data")
    @patch("src.rag.ingestion.generate_legislation_data")
    def test_returns_legislation_with_empty_relationships_when_rel_fails(self, mock_leg, mock_rel):
        legislation = Legislation.model_validate_json(VALID_LEGISLATION_JSON)
        mock_leg.return_value = legislation
        mock_rel.return_value = []

        result = run_ingestion_workflow("some text")

        assert result is not None
        assert result.relationships == []

    @patch("src.rag.ingestion.generate_relationships_data")
    @patch("src.rag.ingestion.generate_legislation_data")
    def test_both_extractions_are_called(self, mock_leg, mock_rel):
        legislation = Legislation.model_validate_json(VALID_LEGISLATION_JSON)
        mock_leg.return_value = legislation
        mock_rel.return_value = []

        run_ingestion_workflow("some text")

        mock_leg.assert_called_once_with("some text")
        mock_rel.assert_called_once_with("some text")