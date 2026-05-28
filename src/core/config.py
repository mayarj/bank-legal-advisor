from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    claude_temperature: float = 0.0
    claude_max_tokens: int = 8096

    database_url: str

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "./data/chromadb"
    chroma_collection: str = "legislation_articles"

    max_critique_retries: int = 2
    graph_k_depth: int = 2
    similarity_n_results: int = 5


settings = Settings()
