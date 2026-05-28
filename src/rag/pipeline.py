import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Relationship as RelationshipModel
from src.db.schemas import Legislation, Relationship
from src.rag.ingestion import run_ingestion_workflow
from src.rag.parser import parse_pdf
from src.rag.vectorstore import add_legislation


async def run_pipeline(file_path: str, session: AsyncSession) -> Legislation | None:
    text = parse_pdf(file_path)

    if not text:
        return None

    legislation = run_ingestion_workflow(text)

    if legislation is None:
        return None

    add_legislation(legislation)
    await _save_relationships(legislation.relationships, session)

    return legislation


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