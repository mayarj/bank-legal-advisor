from src.core.llm import invoke
from src.core.prompts import query_rewrite_prompt
from src.rag.vectorstore import SearchResult, search


def retrieve(
    query: str,
    n_results: int = 5,
    filters: dict | None = None,
    rewrite: bool = True,
) -> list[SearchResult]:
    search_query = _rewrite_query(query) if rewrite else query
    return search(search_query, n_results=n_results, filters=filters)


def retrieve_active(
    query: str,
    n_results: int = 5,
    rewrite: bool = True,
) -> list[SearchResult]:
    return retrieve(query, n_results=n_results, filters={"status": "active"}, rewrite=rewrite)


def _rewrite_query(query: str) -> str:
    system_msg, prompt = query_rewrite_prompt(query)
    return invoke(system_msg, prompt).strip()