import json
import re
from typing import Annotated, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from src.core.config import settings
from src.core.llm import invoke
from src.core.prompts import (
    search_planning_prompt,
    critique_prompt,
    evaluate_relationships_prompt,
    synthesis_prompt,
)


# ── State ─────────────────────────────────────────────────────────────────────

class LegalAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_plan: dict | None
    needs_clarification: bool
    search_results: str
    retrieved_articles: list[str]
    parent_context: str
    needs_children: bool
    children_context: str
    draft_answer: str
    critique_feedback: str
    critique_passed: bool
    critique_iterations: int


# ── Helpers ───────────────────────────────────────────────────────────────────

_ARTICLE_REF_PATTERN = re.compile(r"\[([^\|]+)\s*\|\s*Article\s+(\S+)\s*\|")


def _parse_article_refs(text: str) -> list[tuple[str, str]]:
    """Extract (legislation_code, article_number) pairs from similarity result text."""
    return [(code.strip(), num.strip()) for code, num in _ARTICLE_REF_PATTERN.findall(text)]


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_legal_agent(tools: list[BaseTool]):
    """Build and compile the legal agent graph.
    Tools must come from an active mcp_client() context — keep that context open
    for the full lifetime of agent invocations."""

    tool_map = {t.name: t for t in tools}

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def plan_search(state: LegalAgentState) -> dict:
        system_msg, prompt = search_planning_prompt(state["query"])
        raw = invoke(system_msg, prompt)
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {"suggested_search_queries": [state["query"]], "target_legislation": None}

        ambiguous = (
            not plan.get("search_keywords")
            and not plan.get("target_legislation")
            and not plan.get("suggested_search_queries")
        )
        return {"search_plan": plan, "needs_clarification": ambiguous}

    def ask_clarification(state: LegalAgentState) -> dict:
        clarification = interrupt({
            "message": (
                "Your question is ambiguous. Could you provide more detail?\n"
                f"For example: which legislation, time period, or specific scenario "
                f"are you asking about?\n\nOriginal question: {state['query']}"
            )
        })
        refined_query = f"{state['query']} {clarification}"
        return {"query": refined_query, "needs_clarification": False}

    def run_search(state: LegalAgentState) -> dict:
        plan = state.get("search_plan") or {}
        queries = plan.get("suggested_search_queries") or [state["query"]]
        keywords: list[str] = plan.get("search_keywords") or []
        primary_query = queries[0]

        # Primary: hybrid (semantic + BM25), with first keyword as an exact boost
        hybrid_result = tool_map["hybrid_search_legislation"].invoke({
            "query": primary_query,
            "keyword": keywords[0] if keywords else None,
            "n_results": settings.similarity_n_results,
            "rewrite": True,
        })

        # Secondary: exact search for up to 2 keywords that are specific legal terms
        exact_parts = []
        for kw in keywords[:2]:
            exact_result = tool_map["exact_word_search"].invoke({"keyword": kw, "n_results": 5})
            if exact_result and f"No active articles found containing '{kw}'" not in exact_result:
                exact_parts.append(exact_result)

        if exact_parts:
            combined = hybrid_result + "\n\n--- Exact keyword matches ---\n\n" + "\n\n---\n\n".join(exact_parts)
        else:
            combined = hybrid_result

        return {"search_results": combined}

    def retrieve_articles(state: LegalAgentState) -> dict:
        retrieved = []
        seen: set[tuple[str, str]] = set()
        for legislation_code, article_number in _parse_article_refs(state["search_results"]):
            key = (legislation_code, article_number)
            if key in seen:
                continue
            seen.add(key)
            content = tool_map["get_article_data"].invoke({
                "legislation_code": legislation_code,
                "article_number": article_number,
            })
            retrieved.append(content)

        if not retrieved:
            retrieved = [state["search_results"]]

        return {"retrieved_articles": retrieved}

    async def traverse_parents(state: LegalAgentState) -> dict:
        parent_sections = []
        seen: set[tuple[str, str]] = set()
        for legislation_code, article_number in _parse_article_refs(state["search_results"]):
            key = (legislation_code, article_number)
            if key in seen:
                continue
            seen.add(key)
            context = await tool_map["get_relationship_map"].ainvoke({
                "legislation_code": legislation_code,
                "article_number": article_number,
                "k_depth": settings.graph_k_depth,
                "parents": True,
            })
            if context and context != "No related legislation found.":
                parent_sections.append(
                    f"[Parents of {legislation_code} Article {article_number}]\n{context}"
                )

        return {"parent_context": "\n\n".join(parent_sections) or "No parent relationships found."}

    def evaluate_relationships(state: LegalAgentState) -> dict:
        if state["parent_context"] == "No parent relationships found.":
            return {"needs_children": False}

        system_msg, prompt = evaluate_relationships_prompt(
            state["query"], state["parent_context"]
        )
        raw = invoke(system_msg, prompt)
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            decision = {"legislations_to_retrieve": [], "needs_children": False}

        extra_articles = []
        for code in decision.get("legislations_to_retrieve", []):
            content = tool_map["get_legislation_data"].invoke({"legislation_code": code})
            extra_articles.append(content)

        return {
            "retrieved_articles": state["retrieved_articles"] + extra_articles,
            "needs_children": decision.get("needs_children", False),
        }

    async def traverse_children(state: LegalAgentState) -> dict:
        children_sections = []
        seen: set[tuple[str, str]] = set()
        for legislation_code, article_number in _parse_article_refs(state["search_results"]):
            key = (legislation_code, article_number)
            if key in seen:
                continue
            seen.add(key)
            context = await tool_map["get_relationship_map"].ainvoke({
                "legislation_code": legislation_code,
                "article_number": article_number,
                "k_depth": settings.graph_k_depth,
                "parents": False,
            })
            if context and context != "No related legislation found.":
                children_sections.append(
                    f"[Children of {legislation_code} Article {article_number}]\n{context}"
                )

        return {"children_context": "\n\n".join(children_sections) or "No child relationships found."}

    def synthesize_answer(state: LegalAgentState) -> dict:
        articles_context = "\n\n---\n\n".join(state["retrieved_articles"])
        relationship_context = state["parent_context"]
        if state.get("children_context"):
            relationship_context += f"\n\n{state['children_context']}"

        system_msg, prompt = synthesis_prompt(
            query=state["query"],
            articles_context=articles_context,
            relationship_context=relationship_context,
            critique_feedback=state.get("critique_feedback", ""),
        )
        draft = invoke(system_msg, prompt)
        return {"draft_answer": draft}

    def critique_answer(state: LegalAgentState) -> dict:
        system_msg, prompt = critique_prompt(state["query"], state["draft_answer"])
        raw = invoke(system_msg, prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"passed": True, "feedback": "", "missing_aspects": []}

        iterations = state.get("critique_iterations", 0) + 1
        force_pass = iterations >= settings.max_critique_retries

        return {
            "critique_passed": result.get("passed", True) or force_pass,
            "critique_feedback": result.get("feedback", ""),
            "critique_iterations": iterations,
        }

    # ── Conditional routers ───────────────────────────────────────────────────

    def route_after_plan(state: LegalAgentState) -> str:
        return "ask_clarification" if state["needs_clarification"] else "search"

    def route_after_evaluate(state: LegalAgentState) -> str:
        return "traverse_children" if state["needs_children"] else "synthesize"

    def route_after_critique(state: LegalAgentState) -> str:
        return END if state["critique_passed"] else "synthesize"

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(LegalAgentState)

    graph.add_node("plan", plan_search)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("search", run_search)
    graph.add_node("retrieve_articles", retrieve_articles)
    graph.add_node("traverse_parents", traverse_parents)
    graph.add_node("evaluate_relationships", evaluate_relationships)
    graph.add_node("traverse_children", traverse_children)
    graph.add_node("synthesize", synthesize_answer)
    graph.add_node("critique", critique_answer)

    graph.set_entry_point("plan")

    graph.add_conditional_edges("plan", route_after_plan, {
        "ask_clarification": "ask_clarification",
        "search": "search",
    })
    graph.add_edge("ask_clarification", "search")
    graph.add_edge("search", "retrieve_articles")
    graph.add_edge("retrieve_articles", "traverse_parents")
    graph.add_edge("traverse_parents", "evaluate_relationships")
    graph.add_conditional_edges("evaluate_relationships", route_after_evaluate, {
        "traverse_children": "traverse_children",
        "synthesize": "synthesize",
    })
    graph.add_edge("traverse_children", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {
        END: END,
        "synthesize": "synthesize",
    })

    return graph.compile()