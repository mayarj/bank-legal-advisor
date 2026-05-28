from sentence_transformers import SentenceTransformer
from src.core.config import settings

_model = SentenceTransformer(settings.embedding_model)


def embed(text: str) -> list[float]:
    return _model.encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts, batch_size=32, show_progress_bar=False).tolist()
