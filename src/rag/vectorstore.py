from dataclasses import dataclass
from typing import Optional

import chromadb

from src.core.config import settings
from src.db.schemas import Legislation
from src.rag.embeddings import embed, embed_batch

_client = chromadb.PersistentClient(path=settings.chroma_path)
_collection = _client.get_or_create_collection(
    name=settings.chroma_collection,
    metadata={"hnsw:space": "cosine"},
)


@dataclass
class ArticleResult:
    legislation_code: str
    article_number: str
    content: str
    subject: str
    status: str
    issuer: str
    date: str


@dataclass
class LegislationResult:
    code: str
    subject: str
    status: str
    issuer: str
    date: str
    articles: dict[str, str]  # article_number → content


@dataclass
class SearchResult:
    article_id: str
    legislation_code: str
    article_number: str
    content: str
    subject: str
    status: str
    distance: float


def add_legislation(legislation: Legislation) -> None:
    if not legislation.articles:
        return

    ids, documents, metadatas = [], [], []

    for article_number, content in legislation.articles.items():
        article_id = f"{legislation.code}_article_{article_number}"
        ids.append(article_id)
        documents.append(content)
        metadatas.append({
            "legislation_code": legislation.code,
            "article_number": article_number,
            "status": legislation.status.value,
            "issuer": legislation.issuer,
            "subject": legislation.subject,
            "date": legislation.date.isoformat(),
        })

    batch_embeddings = embed_batch([
        f"{legislation.subject} | Article {num}: {content}"
        for num, content in legislation.articles.items()
    ])

    _collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=batch_embeddings,
        metadatas=metadatas,
    )


def search(
    query: str,
    n_results: int = 5,
    filters: dict | None = None,
) -> list[SearchResult]:
    query_embedding = embed(query)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i, article_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        output.append(SearchResult(
            article_id=article_id,
            legislation_code=meta["legislation_code"],
            article_number=meta["article_number"],
            content=results["documents"][0][i],
            subject=meta["subject"],
            status=meta["status"],
            distance=results["distances"][0][i],
        ))

    return output


def get_article(legislation_code: str, article_number: str) -> Optional[ArticleResult]:
    article_id = f"{legislation_code}_article_{article_number}"
    result = _collection.get(
        ids=[article_id],
        include=["documents", "metadatas"],
    )

    if not result["ids"]:
        return None

    meta = result["metadatas"][0]
    return ArticleResult(
        legislation_code=meta["legislation_code"],
        article_number=meta["article_number"],
        content=result["documents"][0],
        subject=meta["subject"],
        status=meta["status"],
        issuer=meta["issuer"],
        date=meta["date"],
    )


def get_legislation(legislation_code: str) -> Optional[LegislationResult]:
    result = _collection.get(
        where={"legislation_code": legislation_code},
        include=["documents", "metadatas"],
    )

    if not result["ids"]:
        return None

    meta = result["metadatas"][0]
    articles = {
        m["article_number"]: doc
        for m, doc in zip(result["metadatas"], result["documents"])
    }

    return LegislationResult(
        code=meta["legislation_code"],
        subject=meta["subject"],
        status=meta["status"],
        issuer=meta["issuer"],
        date=meta["date"],
        articles=articles,
    )


def delete_legislation(legislation_code: str) -> None:
    _collection.delete(where={"legislation_code": legislation_code})
