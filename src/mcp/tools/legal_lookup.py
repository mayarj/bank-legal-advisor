from src.mcp.app import mcp
from src.db.session import AsyncSessionFactory
from src.rag.graph import format_for_llm, get_children, get_parents
from src.rag.retriever import retrieve_active, retrieve_active_hybrid, retrieve_exact
from src.rag.vectorstore import ArticleResult, LegislationResult, get_article, get_legislation


@mcp.tool()
def get_article_data(legislation_code: str, article_number: str) -> str:
    """Retrieve a single article by its number and the legislation code it belongs to.
    Returns the article content together with legislation metadata (subject, status, issuer, date).
    Use this when a specific article is needed and the context of surrounding articles is not required."""
    data = get_article(legislation_code, article_number)
    if data is None:
        return f"Article {article_number} not found in legislation {legislation_code}."
    return _format_article(data)


@mcp.tool()
def get_legislation_data(legislation_code: str) -> str:
    """Retrieve the full text of a legislation by its code.
    Returns all articles together with legislation metadata (subject, status, issuer, date).
    Use this when the entire legislation is needed, not just a single article."""
    data = get_legislation(legislation_code)
    if data is None:
        return f"Legislation {legislation_code} not found."
    return _format_legislation(data)


@mcp.tool()
async def get_relationship_map(
    legislation_code: str,
    article_number: str | None = None,
    k_depth: int = 2,
    parents: bool = True,
) -> str:
    """Traverse the legislation relationship graph.
    Set parents=True to discover what legislation or articles affect this one (upstream).
    Set parents=False to discover what this legislation or article affects (downstream).
    Returns each related legislation with its relationship type and the condition under
    which the relationship applies — use the condition to decide whether to retrieve it."""
    async with AsyncSessionFactory() as session:
        if parents:
            nodes = await get_parents(session, legislation_code, article_number, k_depth)
            return format_for_llm(nodes, "parent")
        nodes = await get_children(session, legislation_code, article_number, k_depth)
        return format_for_llm(nodes, "child")


@mcp.tool()
def similarity_search(
    query: str,
    n_results: int = 5,
    rewrite: bool = True,
) -> str:
    """Search for legislation articles semantically relevant to a question or topic.
    Only searches articles currently in force (repealed/superseded articles are excluded;
    amended articles are still in force and are included).
    Set rewrite=True (default) to let the LLM rephrase the query into legal terminology
    before searching — improves accuracy for conversational questions."""
    results = retrieve_active(query, n_results=n_results, rewrite=rewrite)
    if not results:
        return "No relevant legislation found for the given query."
    return _format_search_results(results)


@mcp.tool()
def hybrid_search_legislation(
    query: str,
    keyword: str | None = None,
    n_results: int = 5,
    rewrite: bool = True,
) -> str:
    """Search legislation using semantic + BM25 hybrid ranking (Reciprocal Rank Fusion).
    More robust than pure semantic search — surfaces articles that rank highly under
    multiple signals (meaning and term frequency).
    Optionally provide a keyword to also boost articles that contain it verbatim."""
    results = retrieve_active_hybrid(query, keyword=keyword, n_results=n_results, rewrite=rewrite)
    if not results:
        return "No relevant legislation found for the given query."
    return _format_search_results(results)


@mcp.tool()
def exact_word_search(
    keyword: str,
    n_results: int = 10,
) -> str:
    """Find in-force legislation articles that contain an exact word or phrase.
    Use when you need articles mentioning a specific term, article reference, or legal phrase.
    keyword match is case-sensitive and must appear verbatim in the article text."""
    results = retrieve_exact(keyword, n_results=n_results)
    if not results:
        return f"No active articles found containing '{keyword}'."
    return _format_search_results(results)


# ── Private formatters ────────────────────────────────────────────────────────

def _format_article(data: ArticleResult) -> str:
    return (
        f"Legislation: {data.legislation_code}\n"
        f"Subject: {data.subject}\n"
        f"Status: {data.status}\n"
        f"Issuer: {data.issuer}\n"
        f"Date: {data.date}\n"
        f"\nArticle {data.article_number}:\n{data.content}"
    )


def _format_legislation(data: LegislationResult) -> str:
    articles_text = "\n\n".join(
        f"Article {num}:\n{content}"
        for num, content in sorted(data.articles.items())
    )
    return (
        f"Legislation: {data.code}\n"
        f"Subject: {data.subject}\n"
        f"Status: {data.status}\n"
        f"Issuer: {data.issuer}\n"
        f"Date: {data.date}\n"
        f"\n{articles_text}"
    )


def _format_search_results(results) -> str:
    parts = []
    for r in results:
        parts.append(
            f"[{r.legislation_code} | Article {r.article_number} | {r.status}]\n"
            f"Subject: {r.subject}\n"
            f"{r.content}"
        )
    return "\n\n---\n\n".join(parts)