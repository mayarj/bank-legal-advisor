"""LangGraph checkpointer wiring.

The checkpointer persists agent state between turns so a clarification interrupt
can pause on one request and resume on the next. Use the ``postgres`` backend to
make that state durable and shared across backend replicas; ``memory`` keeps it
in-process (fine for local development, lost on restart).
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver

from src.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """Yield a checkpointer for the application's lifetime.

    For the Postgres backend this opens a shared async connection pool and
    ensures the checkpoint tables exist (idempotent), closing the pool on exit.
    """
    backend = settings.checkpointer_backend.lower()

    if backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            conninfo=settings.psycopg_dsn,
            max_size=settings.checkpoint_pool_max_size,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        await pool.open()
        try:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()  # create checkpoint tables if missing (idempotent)
            logger.info("Checkpointer: Postgres (durable, shared)")
            yield saver
        finally:
            await pool.close()
        return

    from langgraph.checkpoint.memory import MemorySaver

    logger.warning(
        "Checkpointer: in-memory (CHECKPOINTER_BACKEND=%s). Agent resume state is "
        "not durable and is not shared across replicas.",
        settings.checkpointer_backend,
    )
    yield MemorySaver()