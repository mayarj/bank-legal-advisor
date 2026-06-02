"""Status reconciliation — the single idempotent operation that keeps article status
correct as legislation is ingested in any order.

`recompute_article_status` re-reads *all* of an article's incoming relationships every
time and folds them, so it does not matter whether the affecting law or the affected law
was ingested first. `reconcile_after_ingest` simply calls it for every article the newly
ingested legislation touches, in both directions. See STATUS_PROPAGATION_PLAN.md.
"""

from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from src.db import crud
from src.db.models import Article as ArticleModel
from src.db.schemas import ArticleStatus, Legislation
from src.rag.status_policy import article_baseline, fold_status
from src.rag.vectorstore import update_article_status

# Effects whose affecting legislation has not been ingested yet have no known date.
# They sort earliest so any dated effect outranks them — they should not win the fold.
_UNKNOWN_DATE = date.min


async def recompute_article_status(
    session: AsyncSession,
    legislation_code: str,
    article_number: str,
) -> ArticleModel | None:
    """Recompute one article's (status, is_in_force) from its baseline + every incoming
    relationship, then persist to Postgres and sync the ChromaDB metadata. Idempotent.
    Returns None if the article does not exist yet (e.g. a forward reference to a law
    not yet ingested) — nothing to update."""
    article = await crud.get_article_row(session, legislation_code, article_number)
    if article is None:
        return None

    legislation = await crud.get_legislation_row(session, legislation_code)
    baseline = (
        article_baseline(legislation.status)
        if legislation is not None
        else (ArticleStatus.ACTIVE, True)
    )

    relationships = await crud.get_incoming_relationships(
        session, legislation_code, article_number
    )
    incoming: list[tuple[date, object]] = []
    for rel in relationships:
        father = await crud.get_legislation_row(session, rel.father_legislation)
        effect_date = father.date if father is not None else _UNKNOWN_DATE
        incoming.append((effect_date, rel.type))

    status, is_in_force = fold_status(baseline, incoming)

    article.status = status
    article.is_in_force = is_in_force
    await session.flush()

    # Postgres is the source of truth; mirror into the vector store for search filtering.
    update_article_status(article.id, status.value, is_in_force)
    return article


async def reconcile_after_ingest(
    session: AsyncSession, legislation: Legislation
) -> None:
    """Recompute every article affected by ingesting `legislation`, in both directions:
      - forward: this legislation's own articles, which prior laws may already affect;
      - backward: articles in other legislation that this one affects.
    """
    affected: set[tuple[str, str]] = {
        (legislation.code, number) for number in legislation.articles
    }

    for rel in legislation.relationships:
        if rel.affected_article is not None:
            affected.add((rel.affected_legislation, rel.affected_article))
        else:
            # Whole-legislation effect → applies to every article that already exists.
            for art in await crud.get_articles_for_legislation(
                session, rel.affected_legislation
            ):
                affected.add((art.legislation_code, art.article_number))

    for code, number in affected:
        await recompute_article_status(session, code, number)