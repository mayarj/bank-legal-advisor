from dataclasses import dataclass, replace as dc_replace
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

from src.core.config import settings
from src.db.schemas import Legislation
from src.rag.embeddings import embed, embed_batch
from src.rag.status_policy import article_baseline

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
    score: float = 0.0  # RRF score populated by hybrid_search


class _BM25Index:
    """In-memory BM25 index, lazily populated from ChromaDB on first query."""

    def __init__(self) -> None:
        self._entries: list[tuple[str, str, dict]] = []  # (id, content, metadata)
        self._bm25: BM25Okapi | None = None
        self._loaded = False

    def _ensure_ready(self) -> None:
        if not self._loaded:
            self._load_from_chroma()
            self._loaded = True
        if self._bm25 is None and self._entries:
            self._rebuild()

    def _load_from_chroma(self) -> None:
        result = _collection.get(include=["documents", "metadatas"])
        self._entries = [
            (aid, doc, meta)
            for aid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def _rebuild(self) -> None:
        corpus = [content.lower().split() for _, content, _ in self._entries]
        self._bm25 = BM25Okapi(corpus)

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        if not self._loaded:
            return  # lazy load will include these from ChromaDB when first queried
        existing = {entry[0] for entry in self._entries}
        new_entries = [
            (aid, doc, meta)
            for aid, doc, meta in zip(ids, documents, metadatas)
            if aid not in existing
        ]
        if new_entries:
            self._entries.extend(new_entries)
            self._bm25 = None  # mark for rebuild on next search

    def update_metadata(self, article_id: str, metadata: dict) -> None:
        """Refresh the cached metadata for one article (e.g. after a status change) so
        filtered BM25 searches see the new value. Content is unchanged, so the BM25 scores
        do not need rebuilding."""
        if not self._loaded:
            return
        for i, (aid, content, _meta) in enumerate(self._entries):
            if aid == article_id:
                self._entries[i] = (aid, content, metadata)
                break

    def search(
        self,
        query: str,
        n_results: int,
        filters: dict | None = None,
    ) -> list[tuple[float, str, str, dict]]:  # (score, id, content, metadata)
        self._ensure_ready()
        if not self._entries or self._bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        ranked = [
            (score, article_id, content, meta)
            for score, (article_id, content, meta) in zip(scores, self._entries)
            if score > 0
            and (filters is None or all(meta.get(k) == v for k, v in filters.items()))
        ]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:n_results]


_bm25_index = _BM25Index()


def add_legislation(legislation: Legislation) -> None:
    if not legislation.articles:
        return

    # Seed article metadata from the legislation baseline; reconciliation overwrites
    # status/is_in_force afterwards for any article that relationships actually affect.
    base_status, base_in_force = article_baseline(legislation.status)

    ids, documents, metadatas = [], [], []

    for article_number, content in legislation.articles.items():
        ids.append(f"{legislation.code}_article_{article_number}")
        documents.append(content)
        metadatas.append({
            "legislation_code": legislation.code,
            "article_number": article_number,
            "status": base_status.value,
            "is_in_force": base_in_force,
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
    _bm25_index.add(ids, documents, metadatas)


def update_article_status(article_id: str, status: str, is_in_force: bool) -> None:
    """Mirror a recomputed article status into ChromaDB metadata. Best-effort: a no-op if
    the article is not in the vector store (Postgres remains the source of truth). Existing
    metadata fields (subject, issuer, …) are preserved."""
    existing = _collection.get(ids=[article_id], include=["metadatas"])
    if not existing["ids"]:
        return

    meta = dict(existing["metadatas"][0])
    meta["status"] = status
    meta["is_in_force"] = is_in_force

    _collection.update(ids=[article_id], metadatas=[meta])
    _bm25_index.update_metadata(article_id, meta)


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


def bm25_search(
    query: str,
    n_results: int = 5,
    filters: dict | None = None,
) -> list[SearchResult]:
    raw = _bm25_index.search(query, n_results, filters)
    return [
        SearchResult(
            article_id=article_id,
            legislation_code=meta["legislation_code"],
            article_number=meta["article_number"],
            content=content,
            subject=meta["subject"],
            status=meta["status"],
            distance=0.0,
            score=bm25_score,
        )
        for bm25_score, article_id, content, meta in raw
    ]


def exact_search(
    keyword: str,
    n_results: int = 5,
    filters: dict | None = None,
) -> list[SearchResult]:
    """Return articles whose text contains keyword as a substring (case-sensitive)."""
    kwargs: dict = {
        "where_document": {"$contains": keyword},
        "limit": n_results,
        "include": ["documents", "metadatas"],
    }
    if filters:
        kwargs["where"] = filters

    result = _collection.get(**kwargs)

    return [
        SearchResult(
            article_id=article_id,
            legislation_code=meta["legislation_code"],
            article_number=meta["article_number"],
            content=content,
            subject=meta["subject"],
            status=meta["status"],
            distance=0.0,
            score=1.0,
        )
        for article_id, content, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]


def hybrid_search(
    query: str,
    keyword: str | None = None,
    n_results: int = 5,
    filters: dict | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion over semantic + BM25 + (optional) exact results."""
    _RRF_K = 60
    candidate_n = n_results * 3

    result_lists: list[list[SearchResult]] = [
        search(query, n_results=candidate_n, filters=filters),
        bm25_search(query, n_results=candidate_n, filters=filters),
    ]
    if keyword:
        result_lists.append(exact_search(keyword, n_results=candidate_n, filters=filters))

    rrf_scores: dict[str, float] = {}
    all_by_id: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, r in enumerate(results):
            rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            all_by_id.setdefault(r.article_id, r)

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:n_results]
    return [
        dc_replace(all_by_id[aid], score=rrf_scores[aid])
        for aid in sorted_ids
        if aid in all_by_id
    ]


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