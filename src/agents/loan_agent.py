import json
import re
from typing import Annotated, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from src.core.llm import invoke
from src.core.prompts import (
    loan_assessment_plan_prompt,
    loan_assessment_synthesis_prompt,
)
from src.agents.legal_agent import _parse_article_refs, build_legal_agent


_CUSTOMER_ID_PATTERN = re.compile(
    r"Customer Profile ID:\s*"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_RISK_LEVEL_PATTERN = re.compile(
    r"Risk Level:\s*\*{0,2}(low|medium|high)\*{0,2}",
    re.IGNORECASE,
)


# ── State ──────────────────────────────────────────────────────────────────────

class LoanAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    loan_id: str
    loan_details: str          # raw text from get_loan_details MCP tool
    customer_context: str      # profile + payment history + existing loans (empty if none)
    assessment_plan: dict | None
    clarification_given: bool  # prevents re-looping after one user clarification
    user_clarification: str
    legal_context: str         # compiled Q&A from legal agent
    cited_articles: list[str]  # collected from legal answers + final synthesis
    final_assessment: str
    risk_level: str            # low | medium | high
    assessment_saved: bool


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_loan_agent(tools: list[BaseTool], checkpointer=None):
    """Build and compile the loan assessment agent graph.

    Tools must come from an active mcp_client() context.
    Pass a shared checkpointer (e.g. AsyncPostgresSaver) so the interrupt-based
    clarification step can suspend on one request and resume on the next.
    Pass checkpointer=False to disable persistence (single-shot, no human-in-the-loop).
    """
    tool_map = {t.name: t for t in tools}

    # ── Nodes ─────────────────────────────────────────────────────────────────

    async def load_loan(state: LoanAgentState) -> dict:
        result = await tool_map["get_loan_details"].ainvoke({"loan_id": state["loan_id"]})
        return {"loan_details": result}

    async def fetch_customer_context(state: LoanAgentState) -> dict:
        """If the loan is linked to a CustomerProfile, retrieve their full context."""
        match = _CUSTOMER_ID_PATTERN.search(state["loan_details"])
        if not match:
            return {"customer_context": ""}

        cid = match.group(1)
        profile  = await tool_map["get_customer_profile_by_id"].ainvoke({"customer_id": cid})
        payments = await tool_map["get_customer_payment_history"].ainvoke({"customer_id": cid})
        loans    = await tool_map["get_customer_loans"].ainvoke({"customer_id": cid})

        return {
            "customer_context": (
                f"=== Customer Profile ===\n{profile}\n\n"
                f"=== Payment History ===\n{payments}\n\n"
                f"=== Existing Loans ===\n{loans}"
            )
        }

    def plan_assessment(state: LoanAgentState) -> dict:
        """LLM analyses the application and produces a structured plan:
        creditworthiness notes, legal questions, and whether user clarification is needed."""
        system_msg, prompt = loan_assessment_plan_prompt(
            loan_details=state["loan_details"],
            customer_context=state.get("customer_context", ""),
            user_clarification=state.get("user_clarification", ""),
        )
        raw = invoke(system_msg, prompt)
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {
                "creditworthiness_notes": raw,
                "risk_indicators": [],
                "legal_questions": [],
                "needs_user_clarification": False,
                "clarification_question": "",
            }
        return {"assessment_plan": plan}

    def ask_clarification(state: LoanAgentState) -> dict:
        """Suspend the graph and surface a precise question to the bank officer."""
        question = (
            state["assessment_plan"].get("clarification_question")
            or "Please provide additional details about this loan application."
        )
        answer = interrupt({"message": question})
        return {
            "user_clarification": str(answer),
            "clarification_given": True,
        }

    async def consult_legal_agent(state: LoanAgentState) -> dict:
        """Run the legal agent once per legal question and compile the answers."""
        questions = (state.get("assessment_plan") or {}).get("legal_questions", [])
        if not questions:
            return {"legal_context": "", "cited_articles": list(state.get("cited_articles") or [])}

        # Non-interactive: this sub-agent runs unattended, so it must never pause
        # for user clarification — ambiguous relationships are simply skipped.
        legal_agent = build_legal_agent(tools, interactive=False)
        parts: list[str] = []
        cited: list[str] = list(state.get("cited_articles") or [])

        for question in questions:
            result = await legal_agent.ainvoke({"query": question, "messages": []})
            answer = result.get("draft_answer", "No answer returned.")
            parts.append(f"Legal Question: {question}\n\nAnswer:\n{answer}")
            for code, num in _parse_article_refs(answer):
                ref = f"{code} | Article {num}"
                if ref not in cited:
                    cited.append(ref)

        return {
            "legal_context": "\n\n---\n\n".join(parts),
            "cited_articles": cited,
        }

    def synthesize_assessment(state: LoanAgentState) -> dict:
        """LLM writes the complete final assessment from all accumulated context."""
        system_msg, prompt = loan_assessment_synthesis_prompt(
            loan_details=state["loan_details"],
            customer_context=state.get("customer_context", ""),
            assessment_plan=state.get("assessment_plan") or {},
            legal_context=state.get("legal_context", ""),
            user_clarification=state.get("user_clarification", ""),
        )
        assessment = invoke(system_msg, prompt)

        match = _RISK_LEVEL_PATTERN.search(assessment)
        risk_level = match.group(1).lower() if match else "medium"

        cited = list(state.get("cited_articles") or [])
        for code, num in _parse_article_refs(assessment):
            ref = f"{code} | Article {num}"
            if ref not in cited:
                cited.append(ref)

        return {
            "final_assessment": assessment,
            "risk_level": risk_level,
            "cited_articles": cited,
        }

    async def save_result(state: LoanAgentState) -> dict:
        """Persist the assessment via the save_assessment MCP tool."""
        # Use the second line of loan_details (Applicant: ...) as the label
        lines = state["loan_details"].split("\n")
        label = f"Loan assessment — {lines[1].strip()}" if len(lines) > 1 else "Loan assessment"

        result = await tool_map["save_assessment"].ainvoke({
            "loan_id": state["loan_id"],
            "legal_question": label,
            "answer": state["final_assessment"],
            "risk_level": state["risk_level"],
            "cited_articles": json.dumps(state.get("cited_articles") or []),
        })
        return {
            "assessment_saved": True,
            "messages": [{"role": "assistant", "content": result}],
        }

    # ── Conditional routers ───────────────────────────────────────────────────

    def route_after_plan(state: LoanAgentState) -> str:
        plan = state.get("assessment_plan") or {}
        if plan.get("needs_user_clarification") and not state.get("clarification_given"):
            return "ask_clarification"
        if plan.get("legal_questions"):
            return "consult_legal_agent"
        return "synthesize"

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(LoanAgentState)

    graph.add_node("load_loan",             load_loan)
    graph.add_node("fetch_customer_context", fetch_customer_context)
    graph.add_node("plan_assessment",        plan_assessment)
    graph.add_node("ask_clarification",      ask_clarification)
    graph.add_node("consult_legal_agent",    consult_legal_agent)
    graph.add_node("synthesize_assessment",  synthesize_assessment)
    graph.add_node("save_result",            save_result)

    graph.set_entry_point("load_loan")

    graph.add_edge("load_loan",             "fetch_customer_context")
    graph.add_edge("fetch_customer_context", "plan_assessment")
    graph.add_conditional_edges("plan_assessment", route_after_plan, {
        "ask_clarification":   "ask_clarification",
        "consult_legal_agent": "consult_legal_agent",
        "synthesize":          "synthesize_assessment",
    })
    # After user provides clarification: re-plan once (clarification_given prevents another loop)
    graph.add_edge("ask_clarification",  "plan_assessment")
    graph.add_edge("consult_legal_agent", "synthesize_assessment")
    graph.add_edge("synthesize_assessment", "save_result")
    graph.add_edge("save_result", END)

    return graph.compile(checkpointer=checkpointer)