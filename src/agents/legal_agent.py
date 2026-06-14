import json
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send, interrupt

from src.core.config import settings
from src.core.llm import invoke
from src.core.prompts import (
    search_planning_prompt,
    critique_prompt,
    assess_article_relationships_prompt,
    select_relationships_prompt,
    synthesis_prompt,
)


# ── State ─────────────────────────────────────────────────────────────────────

class LegalAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_plan: dict | None
    needs_clarification: bool
    search_results: str
    article_refs: list[tuple[str, str]]
    # Fan-out channels — each per-article node appends to these.
    retrieved_articles: Annotated[list[str], operator.add]
    relationship_maps: Annotated[list[str], operator.add]
    ambiguous_relations: Annotated[list[dict], operator.add]
    relationship_clarification: str
    draft_answer: str
    critique_feedback: str
    critique_passed: bool
    critique_iterations: int


_NO_RELATIONS = "No related legislation found."

# ── Helpers ───────────────────────────────────────────────────────────────────

_ARTICLE_REF_PATTERN = re.compile(r"\[([^\|]+)\s*\|\s*Article\s+(\S+)\s*\|")


def _parse_article_refs(text: str) -> list[tuple[str, str]]:
    """Extract (legislation_code, article_number) pairs from similarity result text."""
    return [(code.strip(), num.strip()) for code, num in _ARTICLE_REF_PATTERN.findall(text)]


def _dedup_refs(text: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    refs: list[tuple[str, str]] = []
    for ref in _parse_article_refs(text):
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_legal_agent(tools: list[BaseTool], checkpointer=None, interactive: bool = True):
    """Build and compile the legal agent graph.
    Tools must come from an active mcp_client() context — keep that context open
    for the full lifetime of agent invocations.
    Pass a checkpointer (e.g. AsyncPostgresSaver) to enable the human-in-the-loop
    clarification interrupts to pause and resume across requests.
    Set interactive=False for unattended use (e.g. when consulted by the loan
    agent): the relationship clarification gate then skips ambiguous items
    instead of pausing for the user."""

    tool_map = {t.name: t for t in tools}

    def _retrieve_doc(legislation_code: str, article_number) -> str:
        """Pull a whole legislation, or one article when article_number is given."""
        if article_number:
            return tool_map["get_article_data"].invoke({
                "legislation_code": legislation_code,
                "article_number": str(article_number),
            })
        return tool_map["get_legislation_data"].invoke({"legislation_code": legislation_code})

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
        """First clarification — fires when the planner could not pin down the query."""
        plan = state.get("search_plan") or {}
        options = [q for q in (plan.get("suggested_search_queries") or []) if q][:4]

        lines = [
            "Your question is a bit broad — I want to search the right legislation.",
            f"\nOriginal question: {state['query']}\n",
        ]
        if options:
            lines.append("Did you mean one of these, or something else?")
            for i, opt in enumerate(options, 1):
                lines.append(f"  {i}. {opt}")
            lines.append("\nReply with a number, or add detail (which legislation, time period, or scenario).")
        else:
            lines.append(
                "Could you say which legislation, time period, or specific scenario you mean?"
            )

        clarification = interrupt({"message": "\n".join(lines), "options": options})
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

        refs = _dedup_refs(combined)
        out = {"search_results": combined, "article_refs": refs}
        # No structured references to fan out on — fall back to the raw search text.
        if not refs:
            out["retrieved_articles"] = [combined]
        return out

    async def relate_article(state: dict) -> dict:
        """Fan-out worker: runs once per reference article from the hybrid search.
        Retrieves the base article, traverses BOTH directions of the relationship
        graph, then decides per-article which related documents to pull now and which
        are ambiguous (deferred to the user). Results merge via the reducer channels."""
        code = state["ref_code"]
        article = state["ref_num"]
        query = state["query"]

        base = tool_map["get_article_data"].invoke({
            "legislation_code": code,
            "article_number": article,
        })

        parents = await tool_map["get_relationship_map"].ainvoke({
            "legislation_code": code,
            "article_number": article,
            "k_depth": settings.graph_k_depth,
            "parents": True,
        })
        children = await tool_map["get_relationship_map"].ainvoke({
            "legislation_code": code,
            "article_number": article,
            "k_depth": settings.graph_k_depth,
            "parents": False,
        })

        map_sections = []
        if parents and parents != _NO_RELATIONS:
            map_sections.append(f"[Upstream — legislation affecting {code} Article {article}]\n{parents}")
        if children and children != _NO_RELATIONS:
            map_sections.append(f"[Downstream — legislation affected by {code} Article {article}]\n{children}")

        # No relationships: nothing to decide, just contribute the base article.
        if not map_sections:
            return {"retrieved_articles": [base]}

        combined_map = "\n\n".join(map_sections)
        system_msg, prompt = assess_article_relationships_prompt(
            query, f"{code} Article {article}", combined_map
        )
        raw = invoke(system_msg, prompt)
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            decision = {"retrieve": [], "ambiguous": []}

        articles = [base]
        for doc in decision.get("retrieve", []):
            doc_code = doc.get("legislation_code")
            if doc_code:
                articles.append(_retrieve_doc(doc_code, doc.get("article_number")))

        ambiguous = [
            {
                "legislation_code": item["legislation_code"],
                "article_number": item.get("article_number"),
                "condition": item.get("condition", ""),
                "question": item.get("question", ""),
                "source": f"{code} Article {article}",
            }
            for item in decision.get("ambiguous", [])
            if item.get("legislation_code")
        ]

        updates: dict = {
            "retrieved_articles": articles,
            "relationship_maps": [f"[Relationships for {code} Article {article}]\n{combined_map}"],
        }
        if ambiguous:
            updates["ambiguous_relations"] = ambiguous
        return updates

    async def clarification_gate(state: LegalAgentState) -> dict:
        """Second clarification — fires only when per-article evaluation left some
        relationship conditions ambiguous. Pauses once, aggregating all ambiguous
        items into one descriptive prompt, then retrieves what the user confirms."""
        pending = state.get("ambiguous_relations") or []
        if not pending or not interactive:
            # Nothing to ask, or running unattended — proceed without the user.
            return {}

        lines = [
            "I found related legislation whose conditions may or may not apply to your situation.",
            "Tell me which apply so I pull only what's relevant:\n",
        ]
        for i, item in enumerate(pending, 1):
            ref = item["legislation_code"]
            if item.get("article_number"):
                ref += f" Article {item['article_number']}"
            question = item.get("question") or item.get("condition", "")
            lines.append(f'  {i}. {ref} — {question}')
            lines.append(f'     (condition: "{item.get("condition", "")}", from {item["source"]})')
        lines.append('\nReply with the numbers/codes that apply (e.g. "1,3"), "all", or "none".')

        selection = interrupt({"message": "\n".join(lines), "options": pending})

        system_msg, prompt = select_relationships_prompt(state["query"], str(selection), pending)
        raw = invoke(system_msg, prompt)
        try:
            chosen = json.loads(raw)
        except json.JSONDecodeError:
            chosen = {"retrieve": []}

        extra = [
            _retrieve_doc(doc["legislation_code"], doc.get("article_number"))
            for doc in chosen.get("retrieve", [])
            if doc.get("legislation_code")
        ]
        return {
            "retrieved_articles": extra,
            "relationship_clarification": str(selection),
        }

    def synthesize_answer(state: LegalAgentState) -> dict:
        retrieved = state.get("retrieved_articles") or [state.get("search_results", "")]
        articles_context = "\n\n---\n\n".join(retrieved)

        maps = state.get("relationship_maps") or []
        relationship_context = "\n\n".join(maps) if maps else _NO_RELATIONS

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
        if interactive and state["needs_clarification"]:
            return "ask_clarification"
        return "search"  # unattended: never pause, search with the query as-is

    def dispatch_relationships(state: LegalAgentState):
        """Fan out one relate_article worker per unique reference article."""
        refs = state.get("article_refs") or []
        if not refs:
            return "synthesize"
        return [
            Send("relate_article", {"query": state["query"], "ref_code": c, "ref_num": n})
            for c, n in refs
        ]

    def route_after_critique(state: LegalAgentState) -> str:
        return END if state["critique_passed"] else "synthesize"

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(LegalAgentState)

    graph.add_node("plan", plan_search)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("search", run_search)
    graph.add_node("relate_article", relate_article)
    graph.add_node("clarification_gate", clarification_gate)
    graph.add_node("synthesize", synthesize_answer)
    graph.add_node("critique", critique_answer)

    graph.set_entry_point("plan")

    graph.add_conditional_edges("plan", route_after_plan, {
        "ask_clarification": "ask_clarification",
        "search": "search",
    })
    graph.add_edge("ask_clarification", "search")
    graph.add_conditional_edges("search", dispatch_relationships, ["relate_article", "synthesize"])
    # All fan-out workers converge here before the (optional) second clarification.
    graph.add_edge("relate_article", "clarification_gate")
    graph.add_edge("clarification_gate", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {
        END: END,
        "synthesize": "synthesize",
    })

    return graph.compile(checkpointer=checkpointer)