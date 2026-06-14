import os
import uuid

# Must be set before any src import — pydantic-settings reads these at module load time.
# Pinned so tests are deterministic regardless of the developer's local .env.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("CHROMA_PATH", "/tmp/test-chromadb")
os.environ.setdefault("MAX_CRITIQUE_RETRIES", "2")
os.environ.setdefault("GRAPH_K_DEPTH", "2")

import pytest
import chromadb
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.rag.vectorstore import _BM25Index

from src.db.base import Base
from src.db.models import Relationship as RelationshipModel
from src.db.schemas import Legislation, LegislationStatus, Relationship, RelationshipType

# ── Shared JSON fixtures ──────────────────────────────────────────────────────

VALID_LEGISLATION_JSON = """{
    "code": "LAW-88-2003",
    "date": "2003-06-15",
    "status": "active",
    "issuer": "Central Bank",
    "subject": "Banking loan collateral requirements and interest regulations",
    "articles": {
        "1": "This law governs collateral requirements for all banking loans issued domestically.",
        "2": "All loans exceeding 50,000 units must be secured by real estate collateral.",
        "3": "Personal loans up to 10,000 units may be issued without collateral at bank discretion."
    }
}"""

VALID_RELATIONSHIPS_JSON = """[
    {
        "type": "amends",
        "father_legislation": "LAW-12-2010",
        "father_article": "3",
        "affected_legislation": "LAW-88-2003",
        "affected_article": "2",
        "illustration": "applies when loan amount exceeds 100,000 units after 2010"
    }
]"""

INVALID_JSON = "{bad json: missing quotes, trailing comma,}"


# ── Pydantic object fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_legislation() -> Legislation:
    return Legislation.model_validate_json(VALID_LEGISLATION_JSON)


@pytest.fixture
def sample_legislation_repealed() -> Legislation:
    leg = Legislation.model_validate_json(VALID_LEGISLATION_JSON)
    leg.code = "LAW-01-1990"
    leg.status = LegislationStatus.REPEALED
    leg.subject = "Old repealed collateral rules for small banks"
    leg.articles = {
        "1": "Repealed regulation on minimum loan thresholds.",
    }
    return leg


# ── ChromaDB in-memory fixture ────────────────────────────────────────────────

@pytest.fixture
def test_collection(monkeypatch):
    client = chromadb.EphemeralClient()
    # EphemeralClient shares in-process state; delete stale data from prior tests
    try:
        client.delete_collection("test_legislation")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="test_legislation",
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr("src.rag.vectorstore._collection", collection)
    monkeypatch.setattr("src.rag.vectorstore._bm25_index", _BM25Index())
    return collection


# ── Async SQLite in-memory session fixture ────────────────────────────────────

@pytest.fixture
async def db_session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        yield session

    await engine.dispose()


# ── Graph test-data helper ────────────────────────────────────────────────────

def make_relationship(
    father: str,
    affected: str,
    rel_type: RelationshipType = RelationshipType.AMENDS,
    father_article: str | None = None,
    affected_article: str | None = None,
    illustration: str = "always applies",
) -> RelationshipModel:
    return RelationshipModel(
        id=uuid.uuid4(),
        type=rel_type,
        father_legislation=father,
        father_article=father_article,
        affected_legislation=affected,
        affected_article=affected_article,
        illustration=illustration,
    )