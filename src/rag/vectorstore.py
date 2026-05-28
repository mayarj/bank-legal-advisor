from dataclasses import dataclass

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

    ids, documents, embeddings, metadatas = [], [], [], []

    for article_number, content in legislation.articles.items():
        article_id = f"{legislation.code}_article_{article_number}"
        text_to_embed = f"{legislation.subject} | Article {article_number}: {content}"

        ids.append(article_id)
        documents.append(content)
        embeddings.append(None)  # filled in batch below
        metadatas.append({
            "legislation_code": legislation.code,
            "article_number": article_number,
            "status": legislation.status.value,
            "issuer": legislation.issuer,
            "subject": legislation.subject,
            "date": legislation.date.isoformat(),
        })

    texts_to_embed = [
        f"{legislation.subject} | Article {num}: {content}"
        for num, content in legislation.articles.items()
    ]
    batch_embeddings = embed_batch(texts_to_embed)

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


def delete_legislation(legislation_code: str) -> None:
    _collection.delete(where={"legislation_code": legislation_code})
