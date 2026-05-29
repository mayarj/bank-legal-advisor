from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import customers, ingest, legal, loans
from src.db.session import create_tables
from src.mcp.client import mcp_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    await create_tables()                          # idempotent schema bootstrap
    async with mcp_client() as tools:
        app.state.tools = tools                    # shared across all requests
        yield
    # ── Shutdown: mcp_client context exits, MCP server subprocess stops ──────


app = FastAPI(
    title="Bank Legal Advisor",
    description=(
        "Loan assessment, legal compliance search, and customer credit profiling "
        "for banking legislation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(legal.router)
app.include_router(loans.router)
app.include_router(customers.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}