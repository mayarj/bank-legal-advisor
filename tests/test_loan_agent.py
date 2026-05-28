"""
Checkpoint: loan-agent — LoanAgent StateGraph covering load, customer fetch,
plan, clarification interrupt, legal consultation, synthesis, and save.
"""
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.loan_agent import LoanAgentState, _CUSTOMER_ID_PATTERN, _RISK_LEVEL_PATTERN, build_loan_agent


# ── Shared test data ──────────────────────────────────────────────────────────

_CID = "11111111-1111-1111-1111-111111111111"
_LID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

LOAN_DETAILS_WITH_CUSTOMER = (
    f"ID: {_LID}\n"
    f"Applicant: Ahmed Ali\n"
    "Type: mortgage | Amount: 250000.00 | Duration: 360 months\n"
    "Interest Rate: 4.75% | Collateral: real_estate | Credit Score: 720\n"
    "Purpose: Purchase residential property\n"
    f"Customer Profile ID: {_CID}\n"
    "Status: pending | Created: 2024-01-01"
)

LOAN_DETAILS_NO_CUSTOMER = (
    f"ID: {_LID}\n"
    "Applicant: Sara Khaled\n"
    "Type: personal | Amount: 20000.00 | Duration: 36 months\n"
    "Interest Rate: 9.50% | Collateral: none | Credit Score: N/A\n"
    "Purpose: Home renovation\n"
    "Customer Profile ID: none\n"
    "Status: pending | Created: 2024-01-01"
)

CUSTOMER_PROFILE_TEXT = f"ID: {_CID}\nName: Ahmed Ali | National ID: EG-001\nCredit Score: 720"
PAYMENT_HISTORY_TEXT  = "Payment summary — On-time: 12 | Late: 1 | Missed: 0 | Partial: 0 | Total: 13"
CUSTOMER_LOANS_TEXT   = "No loans found for customer."

VALID_PLAN_JSON = json.dumps({
    "creditworthiness_notes": "Good credit score of 720, stable income.",
    "risk_indicators": ["High loan amount relative to income"],
    "legal_questions": ["What are the collateral requirements for mortgage loans?"],
    "needs_user_clarification": False,
    "clarification_question": "",
})

PLAN_NEEDS_CLARIFICATION = json.dumps({
    "creditworthiness_notes": "Cannot assess without income verification.",
    "risk_indicators": ["Missing income documentation"],
    "legal_questions": [],
    "needs_user_clarification": True,
    "clarification_question": "Please confirm the applicant's monthly net income.",
})

PLAN_NO_LEGAL_NO_CLARIFY = json.dumps({
    "creditworthiness_notes": "Straightforward small personal loan.",
    "risk_indicators": [],
    "legal_questions": [],
    "needs_user_clarification": False,
    "clarification_question": "",
})

LEGAL_AGENT_RESULT = {
    "draft_answer": (
        "According to [LAW-88-2003 | Article 2 | active], loans exceeding 50,000 "
        "units must be secured by real estate collateral."
    ),
    "critique_passed": True,
}

FINAL_ASSESSMENT = (
    "1. APPLICANT SUMMARY\nAhmed Ali applying for mortgage.\n\n"
    "4. RISK ASSESSMENT\nRisk Level: medium\n\n"
    "6. CITED LEGISLATION\n[LAW-88-2003 | Article 2 | active]"
)

CRITIQUE_PASS_JSON = json.dumps({"passed": True, "feedback": "", "missing_aspects": []})


# ── Tool fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def tools_and_map():
    tool_map = {}
    async_tools = [
        "get_loan_details",
        "get_customer_profile_by_id",
        "get_customer_payment_history",
        "get_customer_loans",
        "save_assessment",
        "get_relationship_map",
        "get_article_data",
        "get_legislation_data",
    ]
    sync_tools = [
        "hybrid_search_legislation",
        "exact_word_search",
    ]

    for name in async_tools:
        m = MagicMock()
        m.name = name
        m.ainvoke = AsyncMock()
        tool_map[name] = m

    for name in sync_tools:
        m = MagicMock()
        m.name = name
        tool_map[name] = m

    # Sensible defaults
    tool_map["get_loan_details"].ainvoke.return_value = LOAN_DETAILS_WITH_CUSTOMER
    tool_map["get_customer_profile_by_id"].ainvoke.return_value = CUSTOMER_PROFILE_TEXT
    tool_map["get_customer_payment_history"].ainvoke.return_value = PAYMENT_HISTORY_TEXT
    tool_map["get_customer_loans"].ainvoke.return_value = CUSTOMER_LOANS_TEXT
    tool_map["save_assessment"].ainvoke.return_value = "Assessment saved."
    tool_map["get_relationship_map"].ainvoke = AsyncMock(return_value="No related legislation found.")
    tool_map["hybrid_search_legislation"].invoke.return_value = (
        "[LAW-88-2003 | Article 2 | active]\nAll loans exceeding 50,000 must be secured."
    )
    tool_map["exact_word_search"].invoke.side_effect = lambda args: (
        f"No active articles found containing '{args['keyword']}'."
    )

    return list(tool_map.values()), tool_map


# ── Regex helpers ─────────────────────────────────────────────────────────────

class TestPatterns:

    def test_customer_id_pattern_matches_valid_uuid(self):
        match = _CUSTOMER_ID_PATTERN.search(LOAN_DETAILS_WITH_CUSTOMER)
        assert match is not None
        assert match.group(1) == _CID

    def test_customer_id_pattern_no_match_when_none(self):
        match = _CUSTOMER_ID_PATTERN.search(LOAN_DETAILS_NO_CUSTOMER)
        assert match is None

    def test_risk_level_pattern_extracts_medium(self):
        text = "4. RISK ASSESSMENT\nRisk Level: medium\n"
        match = _RISK_LEVEL_PATTERN.search(text)
        assert match is not None
        assert match.group(1).lower() == "medium"

    def test_risk_level_pattern_case_insensitive(self):
        match = _RISK_LEVEL_PATTERN.search("Risk Level: HIGH")
        assert match.group(1).lower() == "high"

    def test_risk_level_pattern_matches_bold_markdown(self):
        match = _RISK_LEVEL_PATTERN.search("Risk Level: **low**")
        assert match.group(1).lower() == "low"


# ── Load loan ─────────────────────────────────────────────────────────────────

class TestLoadLoan:

    @patch("src.agents.loan_agent.invoke")
    async def test_calls_get_loan_details_tool(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [PLAN_NO_LEGAL_NO_CLARIFY, FINAL_ASSESSMENT]
        agent = build_loan_agent(tools, checkpointer=False)

        await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t1"}},
        )

        tool_map["get_loan_details"].ainvoke.assert_called_once_with({"loan_id": _LID})

    @patch("src.agents.loan_agent.invoke")
    async def test_loan_details_stored_in_state(self, mock_invoke, tools_and_map):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [PLAN_NO_LEGAL_NO_CLARIFY, FINAL_ASSESSMENT]
        agent = build_loan_agent(tools, checkpointer=False)

        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t2"}},
        )

        assert LOAN_DETAILS_WITH_CUSTOMER in result["loan_details"]


# ── Fetch customer context ────────────────────────────────────────────────────

class TestFetchCustomerContext:

    @patch("src.agents.loan_agent.invoke")
    async def test_fetches_all_three_customer_tools_when_customer_linked(
        self, mock_invoke, tools_and_map
    ):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [PLAN_NO_LEGAL_NO_CLARIFY, FINAL_ASSESSMENT]
        agent = build_loan_agent(tools, checkpointer=False)

        await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t3"}},
        )

        tool_map["get_customer_profile_by_id"].ainvoke.assert_called_once_with({"customer_id": _CID})
        tool_map["get_customer_payment_history"].ainvoke.assert_called_once_with({"customer_id": _CID})
        tool_map["get_customer_loans"].ainvoke.assert_called_once_with({"customer_id": _CID})

    @patch("src.agents.loan_agent.invoke")
    async def test_skips_customer_tools_when_no_customer_id(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        tool_map["get_loan_details"].ainvoke.return_value = LOAN_DETAILS_NO_CUSTOMER
        mock_invoke.side_effect = [PLAN_NO_LEGAL_NO_CLARIFY, FINAL_ASSESSMENT]
        agent = build_loan_agent(tools, checkpointer=False)

        await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t4"}},
        )

        tool_map["get_customer_profile_by_id"].ainvoke.assert_not_called()

    @patch("src.agents.loan_agent.invoke")
    async def test_customer_context_contains_all_sections(self, mock_invoke, tools_and_map):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [PLAN_NO_LEGAL_NO_CLARIFY, FINAL_ASSESSMENT]
        agent = build_loan_agent(tools, checkpointer=False)

        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t5"}},
        )

        ctx = result["customer_context"]
        assert "Customer Profile" in ctx
        assert "Payment History" in ctx
        assert "Existing Loans" in ctx


# ── Plan assessment ───────────────────────────────────────────────────────────

class TestPlanAssessment:

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_plan_parsed_from_llm_output(self, mock_invoke, mock_build_legal, tools_and_map):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT]
        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal
        agent = build_loan_agent(tools, checkpointer=False)

        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t6"}},
        )

        assert result["assessment_plan"]["creditworthiness_notes"] == "Good credit score of 720, stable income."

    @patch("src.agents.loan_agent.invoke")
    async def test_invalid_plan_json_falls_back_gracefully(self, mock_invoke, tools_and_map):
        tools, _ = tools_and_map
        mock_invoke.side_effect = ["{not valid json}", FINAL_ASSESSMENT]
        agent = build_loan_agent(tools, checkpointer=False)

        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t7"}},
        )

        assert result["assessment_plan"]["legal_questions"] == []


# ── Legal agent consultation ──────────────────────────────────────────────────

class TestConsultLegalAgent:

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_legal_agent_called_for_each_question(
        self, mock_invoke, mock_build_legal, tools_and_map
    ):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT, CRITIQUE_PASS_JSON]

        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal

        agent = build_loan_agent(tools, checkpointer=False)
        await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t8"}},
        )

        assert mock_legal.ainvoke.call_count == 1  # one legal question in VALID_PLAN_JSON

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_cited_articles_extracted_from_legal_answers(
        self, mock_invoke, mock_build_legal, tools_and_map
    ):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT, CRITIQUE_PASS_JSON]

        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal

        agent = build_loan_agent(tools, checkpointer=False)
        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t9"}},
        )

        assert any("LAW-88-2003" in ref for ref in result["cited_articles"])

    @patch("src.agents.loan_agent.invoke")
    async def test_skips_legal_consultation_when_no_questions(self, mock_invoke, tools_and_map):
        tools, tool_map = tools_and_map
        tool_map["get_loan_details"].ainvoke.return_value = LOAN_DETAILS_NO_CUSTOMER
        mock_invoke.side_effect = [PLAN_NO_LEGAL_NO_CLARIFY, FINAL_ASSESSMENT]
        agent = build_loan_agent(tools, checkpointer=False)

        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t10"}},
        )

        assert result.get("legal_context", "") == ""


# ── Synthesis ─────────────────────────────────────────────────────────────────

class TestSynthesizeAssessment:

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_final_assessment_stored(self, mock_invoke, mock_build_legal, tools_and_map):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT, CRITIQUE_PASS_JSON]
        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal

        agent = build_loan_agent(tools, checkpointer=False)
        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t11"}},
        )

        assert result["final_assessment"] == FINAL_ASSESSMENT

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_risk_level_extracted_from_assessment(
        self, mock_invoke, mock_build_legal, tools_and_map
    ):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT, CRITIQUE_PASS_JSON]
        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal

        agent = build_loan_agent(tools, checkpointer=False)
        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t12"}},
        )

        assert result["risk_level"] == "medium"

    @patch("src.agents.loan_agent.invoke")
    async def test_defaults_risk_level_to_medium_when_not_found(
        self, mock_invoke, tools_and_map
    ):
        tools, tool_map = tools_and_map
        tool_map["get_loan_details"].ainvoke.return_value = LOAN_DETAILS_NO_CUSTOMER
        mock_invoke.side_effect = [
            PLAN_NO_LEGAL_NO_CLARIFY,
            "Assessment with no explicit risk level mention.",  # no "Risk Level:" line
        ]
        agent = build_loan_agent(tools, checkpointer=False)

        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t13"}},
        )

        assert result["risk_level"] == "medium"


# ── Save result ───────────────────────────────────────────────────────────────

class TestSaveResult:

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_save_assessment_tool_called(self, mock_invoke, mock_build_legal, tools_and_map):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT, CRITIQUE_PASS_JSON]
        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal

        agent = build_loan_agent(tools, checkpointer=False)
        await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t14"}},
        )

        tool_map["save_assessment"].ainvoke.assert_called_once()
        call_args = tool_map["save_assessment"].ainvoke.call_args[0][0]
        assert call_args["loan_id"] == _LID
        assert call_args["risk_level"] == "medium"

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_assessment_saved_flag_set(self, mock_invoke, mock_build_legal, tools_and_map):
        tools, _ = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT, CRITIQUE_PASS_JSON]
        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal

        agent = build_loan_agent(tools, checkpointer=False)
        result = await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t15"}},
        )

        assert result["assessment_saved"] is True

    @patch("src.agents.loan_agent.build_legal_agent")
    @patch("src.agents.loan_agent.invoke")
    async def test_cited_articles_passed_as_json_to_save_tool(
        self, mock_invoke, mock_build_legal, tools_and_map
    ):
        tools, tool_map = tools_and_map
        mock_invoke.side_effect = [VALID_PLAN_JSON, FINAL_ASSESSMENT, CRITIQUE_PASS_JSON]
        mock_legal = MagicMock()
        mock_legal.ainvoke = AsyncMock(return_value=LEGAL_AGENT_RESULT)
        mock_build_legal.return_value = mock_legal

        agent = build_loan_agent(tools, checkpointer=False)
        await agent.ainvoke(
            {"loan_id": _LID, "messages": []},
            config={"configurable": {"thread_id": "t16"}},
        )

        call_args = tool_map["save_assessment"].ainvoke.call_args[0][0]
        articles = json.loads(call_args["cited_articles"])
        assert isinstance(articles, list)