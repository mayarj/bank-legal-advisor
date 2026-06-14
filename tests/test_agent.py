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
    for name in [
        "hybrid_search_legislation", "exact_word_search",
        "get_article_data", "get_legislation_data", "get_relationship_map",
    ]:
        m = MagicMock()
        m.name = name
        tool_map[name] = m

    tool_map["hybrid_search_legislation"].invoke.return_value = SEARCH_RESULTS_TEXT
    # Default: no exact matches so exact results don't affect article ref parsing
    tool_map["exact_word_search"].invoke.side_effect = lambda args: (
        f"No active articles found containing '{args['keyword']}'."
    )
    tool_map["get_article_data"].invoke.return_value = ARTICLE_TEXT
    tool_map["get_legislation_data"].invoke.return_value = "Full legislation content."
    # Returns the sentinel so relate_article finds no relationships and skips the
    # per-article decision LLM call (no extra invoke beyond plan/synthesize/critique).
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
    async def test_hybrid_search_tool_is_called(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        tool_map["hybrid_search_legislation"].invoke.assert_called_once()

    @patch("src.agents.legal_agent.invoke")
    async def test_exact_word_search_called_for_plan_keywords(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        # VALID_PLAN_JSON has search_keywords: ["loan", "collateral"] — capped at 2
        assert tool_map["exact_word_search"].invoke.call_count == 2

    @patch("src.agents.legal_agent.invoke")
    async def test_get_article_data_called_for_each_ref(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        two_refs = (
            "[LAW-88-2003 | Article 2 | active]\nContent A.\n\n---\n\n"
            "[LAW-12-2010 | Article 5 | active]\nContent B."
        )
        tool_map["hybrid_search_legislation"].invoke.return_value = two_refs
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

        call_kwargs = tool_map["hybrid_search_legislation"].invoke.call_args[0][0]
        assert call_kwargs["query"] == "loan requirements"


# ── With relationships (per-article fan-out) ──────────────────────────────────

class TestAgentWithRelationships:

    @patch("src.agents.legal_agent.invoke")
    async def test_clearly_applicable_relationship_is_retrieved(
        self, mock_invoke, tools_and_map
    ):
        tools, tool_map = tools_and_map
        tool_map["get_relationship_map"].ainvoke = AsyncMock(
            return_value="LAW-12-2010 amends this article — always applies"
        )
        decision_json = json.dumps({
            "retrieve": [{"legislation_code": "LAW-12-2010", "article_number": None}],
            "ambiguous": [],
            "reasoning": "condition always applies",
        })
        mock_invoke.side_effect = [VALID_PLAN_JSON, decision_json, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        # article_number is null → whole legislation pulled via get_legislation_data
        tool_map["get_legislation_data"].invoke.assert_called_once_with(
            {"legislation_code": "LAW-12-2010"}
        )

    @patch("src.agents.legal_agent.invoke")
    async def test_each_article_traverses_both_directions(
        self, mock_invoke, tools_and_map
    ):
        tools, tool_map = tools_and_map
        tool_map["get_relationship_map"].ainvoke = AsyncMock(
            return_value="LAW-01-1990 is an implementing regulation"
        )
        decision_json = json.dumps({"retrieve": [], "ambiguous": [], "reasoning": "none apply"})
        mock_invoke.side_effect = [VALID_PLAN_JSON, decision_json, "Answer.", CRITIQUE_PASS_JSON]
        agent = build_legal_agent(tools)

        await agent.ainvoke({"query": "loan requirements", "messages": []})

        # One reference article → relationship map fetched twice: parents and children
        assert tool_map["get_relationship_map"].ainvoke.call_count == 2
        parent_flags = sorted(
            call.args[0]["parents"] if call.args else call.kwargs["parents"]
            for call in tool_map["get_relationship_map"].ainvoke.call_args_list
        )
        assert parent_flags == [False, True]

    @patch("src.agents.legal_agent.invoke")
    async def test_ambiguous_relationship_pauses_for_clarification(
        self, mock_invoke, tools_and_map
    ):
        from langgraph.checkpoint.memory import MemorySaver

        tools, tool_map = tools_and_map
        tool_map["get_relationship_map"].ainvoke = AsyncMock(
            return_value="LAW-99-2020 applies only to consumer loans"
        )
        decision_json = json.dumps({
            "retrieve": [],
            "ambiguous": [{
                "legislation_code": "LAW-99-2020",
                "article_number": None,
                "condition": "applies only to consumer loans",
                "question": "Is this a consumer loan?",
            }],
            "reasoning": "depends on loan type",
        })
        # plan, per-article decision — then the graph should pause before synthesis
        mock_invoke.side_effect = [VALID_PLAN_JSON, decision_json]
        agent = build_legal_agent(tools, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.ainvoke({"query": "loan requirements", "messages": []}, config=config)

        # Paused at the relationship clarification gate — no answer yet
        assert not result.get("draft_answer")
        snapshot = await agent.aget_state(config)
        interrupts = [i for task in snapshot.tasks for i in getattr(task, "interrupts", [])]
        assert interrupts, "expected the agent to interrupt for clarification"
        assert "LAW-99-2020" in interrupts[0].value["message"]

    @patch("src.agents.legal_agent.invoke")
    async def test_non_interactive_never_pauses_on_ambiguity(
        self, mock_invoke, tools_and_map
    ):
        """Unattended mode (used by the loan agent) must run to completion without
        an interrupt, even when relationship conditions are ambiguous."""
        tools, tool_map = tools_and_map
        tool_map["get_relationship_map"].ainvoke = AsyncMock(
            return_value="LAW-99-2020 applies only to consumer loans"
        )
        decision_json = json.dumps({
            "retrieve": [],
            "ambiguous": [{
                "legislation_code": "LAW-99-2020",
                "article_number": None,
                "condition": "applies only to consumer loans",
                "question": "Is this a consumer loan?",
            }],
            "reasoning": "depends on loan type",
        })
        mock_invoke.side_effect = [VALID_PLAN_JSON, decision_json, "Answer.", CRITIQUE_PASS_JSON]
        # No checkpointer — an interrupt here would raise.
        agent = build_legal_agent(tools, interactive=False)

        result = await agent.ainvoke({"query": "loan requirements", "messages": []})

        assert result["draft_answer"] == "Answer."