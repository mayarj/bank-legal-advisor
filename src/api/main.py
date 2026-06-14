from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.api.routes import customers, ingest, legal, loans
from src.core.checkpointer import lifespan_checkpointer
from src.core.config import settings
from src.db.session import create_tables
from src.mcp.client import mcp_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    await create_tables()                                  # idempotent schema bootstrap
    async with lifespan_checkpointer() as checkpointer, mcp_client() as tools:
        app.state.checkpointer = checkpointer              # durable agent-state store
        app.state.tools = tools                            # shared across all requests
        yield
    # ── Shutdown: checkpointer pool closes, MCP server subprocess stops ───────


OPENAPI_TAGS = [
    {"name": "meta", "description": "Service health and metadata."},
    {"name": "ingest", "description": "Upload legislation (PDF or .txt) into the RAG knowledge base."},
    {
        "name": "legal advisor",
        "description": (
            "Ask legal questions. The agent searches legislation, traverses the "
            "relationship graph, and returns a cited answer — pausing for clarification "
            "when the question or a relationship condition is ambiguous."
        ),
    },
    {"name": "loans", "description": "Loan CRUD and the AI loan-assessment agent."},
    {"name": "customers", "description": "Customer profiles, credit scores, and payment history."},
]


app = FastAPI(
    title=settings.app_name,
    description=(
        "Loan assessment, legal compliance search, and customer credit profiling "
        "for banking legislation.\n\n"
        "Interactive API explorer below — expand an endpoint and use **Try it out** "
        "to send live requests. ReDoc is available at `/redoc`."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    contact={"name": "Bank Legal Advisor", "email": "support@example.com"},
    license_info={"name": "Proprietary"},
    swagger_ui_parameters={"docExpansion": "none", "displayRequestDuration": True},
)

app.include_router(ingest.router)
app.include_router(legal.router)
app.include_router(loans.router)
app.include_router(customers.router)


@app.get("/", include_in_schema=False)
def root():
    """Send visitors straight to the Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "env": settings.app_env, "version": settings.app_version}