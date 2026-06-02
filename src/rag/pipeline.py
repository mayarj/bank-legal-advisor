import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.db import crud
from src.db.models import Relationship as RelationshipModel
from src.db.schemas import Legislation, Relationship
from src.rag.ingestion import run_ingestion_workflow
from src.rag.parser import parse_pdf
from src.rag.reconcile import reconcile_after_ingest
from src.rag.vectorstore import add_legislation


async def run_pipeline(file_path: str, session: AsyncSession) -> Legislation | None:
    text = parse_pdf(file_path)

    if not text:
        return None

    legislation = run_ingestion_workflow(text)

    if legislation is None:
        return None

    add_legislation(legislation)                              # content + embeddings → ChromaDB
    await _save_legislation_metadata(legislation, session)    # legislation + article rows → Postgres
    await _save_relationships(legislation.relationships, session)
    await reconcile_after_ingest(session, legislation)        # propagate status, both directions

    return legislation


async def _save_legislation_metadata(
    legislation: Legislation,
    session: AsyncSession,
) -> None:
    await crud.upsert_legislation(session, legislation)
    await crud.upsert_articles(session, legislation)


async def _save_relationships(
    relationships: list[Relationship],
    session: AsyncSession,
) -> None:
    for rel in relationships:
        db_rel = RelationshipModel(
            id=uuid.uuid4(),
            type=rel.type,
            father_legislation=rel.father_legislation,
            father_article=rel.father_article,
            affected_legislation=rel.affected_legislation,
            affected_article=rel.affected_article,
            illustration=rel.illustration,
        )
        session.add(db_rel)
    await session.flush()