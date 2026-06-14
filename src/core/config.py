from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application metadata ──────────────────────────────────────────────────
    app_name: str = "Bank Legal Advisor"
    app_version: str = "1.0.0"
    app_env: str = "development"  # development | staging | production

    # ── LLM (Claude) ──────────────────────────────────────────────────────────
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    claude_temperature: float = 0.0
    claude_max_tokens: int = 8096

    # ── Primary (application) database ────────────────────────────────────────
    database_url: str

    # ── LangGraph checkpointer (durable agent state for resume) ───────────────
    # "memory"   → in-process MemorySaver (single instance only; lost on restart)
    # "postgres" → shared AsyncPostgresSaver, enabling resume across replicas
    checkpointer_backend: str = "memory"
    # Optional dedicated DSN for the checkpointer; falls back to database_url.
    checkpoint_database_url: str | None = None
    checkpoint_pool_max_size: int = 10

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Vector store (ChromaDB) ───────────────────────────────────────────────
    # "embedded" → local on-disk PersistentClient (single instance)
    # "http"     → shared Chroma server via HttpClient (multi-replica)
    chroma_mode: str = "embedded"
    chroma_path: str = "./data/chromadb"
    chroma_collection: str = "legislation_articles"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False

    # ── Lexical (BM25) index ──────────────────────────────────────────────────
    # Seconds between cheap cross-process staleness checks against the shared
    # vector store. The in-memory BM25 index reloads when the shared corpus
    # changes. Set < 0 to disable periodic refresh (single-instance only).
    lexical_refresh_seconds: float = 30.0

    # ── Agent behaviour ───────────────────────────────────────────────────────
    max_critique_retries: int = 2
    graph_k_depth: int = 2
    similarity_n_results: int = 5

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def psycopg_dsn(self) -> str:
        """libpq/psycopg3 DSN for the checkpointer, derived from the configured
        URL by stripping any SQLAlchemy async-driver suffix."""
        url = self.checkpoint_database_url or self.database_url
        for suffix in ("+asyncpg", "+psycopg2", "+psycopg"):
            url = url.replace(suffix, "")
        return url


settings = Settings()