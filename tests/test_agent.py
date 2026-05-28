import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.legal_agent import _parse_article_refs, build_legal_agent


# ── Shared test data ──────────────────────────────────────────────────────────

SEARCH_RESULTS_TEXT = (
    "[LAW-88-2003 | Article 2 | active]\n"
    "Subject: Banking collateral\n"
    "All loans exceeding 50,000 units must be secured."
)

ARTICLE_TEXT = (
    "Legislation: LAW-88-2003\nStatus: active\n\n"
    "Article 2:\nAll loans exceeding 50,000 units must be secured."
)

VALID_PLAN_JSON = json.dumps({
    "primary_intent": "find_requirements",
    "target_legislation": "LAW-88-2003",
    "search_keywords": ["loan", "collateral"],
    "suggested_search_queries": ["loan collateral requirements"],
})

CRITIQUE_PASS_JSON = json.dumps({"passed": True, "feedback": "", "missing_aspects": []})
CRITIQUE_FAIL_JSON = json.dumps({
    "passed": False,
    "feedback": "Missing analysis of article 3.",
    "missing_aspects": ["article 3"],
})


@pytest.fixture
def tools_and_map():
    """Mock tools with sensible defaults for the no-parent-context path."""
    tool_map = {}
    for name in ["similarity_search", "get_article_data", "get_legislation_data", "get_relationship_map"]:
        m = MagicMock()
        m.name = name
        tool_map[name] = m

    tool_map["similarity_search"].invoke.return_value = SEARCH_RESULTS_TEXT
    tool_map["get_article_data"].invoke.return_value = ARTICLE_TEXT
    tool_map["get_legislation_data"].invoke.return_value = "Full legislation content."
    # Returns the sentinel that makes traverse_parents skip adding to parent_sections,
    # so evaluate_relationships never calls invoke (short-circuits to needs_children=False)
    tool_map["get_relationship_map"].ainvoke = AsyncMock(
        return_value="No related legislation found."
    )

    return list(tool_map.values()), tool_map


# ── _parse_article_refs ───────────────────────────────────────────────────────

class TestParseArticleRefs:

    def test_extracts_single_reference(self):
        text = "[LAW-88-2003 | Article 2 | active]"
        assert _parse_article_refs(text) == [("LAW-88-2003", "2")]

    def test_extracts_multiple_references(self):
        text = (
            "[LAW-88-2003 | Article 2 | active]\n\n---\n\n"
            "[LAW-12-2010 | Article 5 | active]"
        )
        result = _parse_article_refs(text)
        assert len(result) == 2
        assert ("LAW-88-2003", "2") in result
        assert ("LAW-12-2010", "5") in result

    def test_returns_empty_list_for_no_matches(self):
        assert _parse_article_refs("No legislation found for the given query.") == []

    def test_strips_whitespace_from_code_and_number(self):
        text = "[  LAW-88-2003  | Article  2  | active]"
        assert _parse_article_refs(text) == [("LAW-88-2003", "2")]

    def test_handles_empty_string(self):
        assert _parse_article_refs("") == []


# ── Happy path ────────────────────────────────────────────────────────────────

class TestAgentHappyPath:

    @patch("src.agents.legal_agent.invoke")
    async def test_returns_draft_answer_and_passes_critique(
        self, mock_invoke, tools_and_map
    ):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, "The collateral requirement is...", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        result = await agent.ainvoke({
            "query": "What are collateral requirements for loans?",
            "messages": [],
        })

        assert result["draft_answer"] == "The collateral requirement is..."
        assert result["critique_passed"] is True

    @patch("src.agents.legal_agent.invoke")
    async def test_similarity_search_tool_is_called(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        tool_map["similarity_search"].invoke.assert_called_once()

    @patch("src.agents.legal_agent.invoke")
    async def test_get_article_data_called_for_each_ref(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        two_refs = (
            "[LAW-88-2003 | Article 2 | active]\nContent A.\n\n---\n\n"
            "[LAW-12-2010 | Article 5 | active]\nContent B."
        )
        tool_map["similarity_search"].invoke.return_value = two_refs
        mock_invoke.side_effect = [VALID_PLAN_JSON, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        assert tool_map["get_article_data"].invoke.call_count == 2

    @patch("src.agents.legal_agent.invoke")
    async def test_critique_iterations_increments_to_one(self, mock_invoke, tools_and_map):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        result = await agent.ainvoke({"query": "loan requirements", "messages": []})

        assert result["critique_iterations"] == 1


# ── Critique retry loop ───────────────────────────────────────────────────────

class TestAgentCritiqueRetry:

    @patch("src.agents.legal_agent.invoke")
    async def test_retries_synthesis_after_failed_critique(
        self, mock_invoke, tools_and_map
    ):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [
            VALID_PLAN_JSON,
            "Draft 1.",
            CRITIQUE_FAIL_JSON,   # fails → back to synthesize
            "Draft 2.",
            CRITIQUE_PASS_JSON,
        ]
        agent = build_legal_agent(tools)

        result = await agent.ainvoke({"query": "loan requirements", "messages": []})

        assert result["draft_answer"] == "Draft 2."
        assert result["critique_iterations"] == 2

    @patch("src.agents.legal_agent.invoke")
    async def test_force_passes_after_max_critique_retries(
        self, mock_invoke, tools_and_map
    ):
        # max_critique_retries=2: iteration 2 triggers force_pass regardless of critique
        tools, _ = tools_and_map
        mock_invoke.side_effect = [
            VALID_PLAN_JSON,
            "Draft 1.",
            CRITIQUE_FAIL_JSON,   # iteration 1 — retry
            "Draft 2.",
            CRITIQUE_FAIL_JSON,   # iteration 2 — force_pass
        ]
        agent = build_legal_agent(tools)

        result = await agent.ainvoke({"query": "loan requirements", "messages": []})

        assert result["critique_passed"] is True
        assert result["critique_iterations"] == 2

    @patch("src.agents.legal_agent.invoke")
    async def test_critique_feedback_reaches_second_synthesis(
        self, mock_invoke, tools_and_map
    ):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [
            VALID_PLAN_JSON,
            "Draft 1.",
            CRITIQUE_FAIL_JSON,
            "Draft 2.",
            CRITIQUE_PASS_JSON,
        ]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        # 4th invoke call (index 3) is the second synthesis — its prompt contains the feedback
        second_synthesis_prompt = mock_invoke.call_args_list[3].args[1]
        assert "Missing analysis of article 3." in second_synthesis_prompt


# ── Plan fallback ─────────────────────────────────────────────────────────────

class TestAgentPlanFallback:

    @patch("src.agents.legal_agent.invoke")
    async def test_invalid_plan_json_uses_original_query_as_search(
        self, mock_invoke, tools_and_map
    ):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [
            "{not valid json}",   # plan fails to parse → fallback plan
            "Answer.",
            CRITIQUE_PASS_JSON,
        ]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        call_kwargs = tool_map["similarity_search"].invoke.call_args[0][0]
        assert call_kwargs["query"] == "loan requirements"


# ── With parent relationships ─────────────────────────────────────────────────

class TestAgentWithParents:

    @patch("src.agents.legal_agent.invoke")
    async def test_evaluate_relationships_retrieves_flagged_legislation(
        self, mock_invoke, tools_and_map
    ):
        tools, tool_map = tools_and_map
        tool_map["get_relationship_map"].ainvoke = AsyncMock(
            return_value="LAW-12-2010 amends this article — always applies"
        )
        evaluate_json = json.dumps({
            "legislations_to_retrieve": ["LAW-12-2010"],
            "needs_children": False,
            "reasoning": "directly relevant",
        })
        mock_invoke.side_effect = [VALID_PLAN_JSON, evaluate_json, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        tool_map["get_legislation_data"].invoke.assert_called_once_with(
            {"legislation_code": "LAW-12-2010"}
        )

    @patch("src.agents.legal_agent.invoke")
    async def test_traverse_children_called_when_needs_children_true(
        self, mock_invoke, tools_and_map
    ):
        tools, tool_map = tools_and_map
        tool_map["get_relationship_map"].ainvoke = AsyncMock(
            return_value="LAW-01-1990 is an implementing regulation"
        )
        evaluate_json = json.dumps({
            "legislations_to_retrieve": [],
            "needs_children": True,
            "reasoning": "implementing regulations exist",
        })
        mock_invoke.side_effect = [VALID_PLAN_JSON, evaluate_json, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        # traverse_parents calls it once, traverse_children calls it again
        assert tool_map["get_relationship_map"].ainvoke.call_count == 2