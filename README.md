# Bank Legal Advisor

An AI-powered backend system for banking legal compliance and loan assessment. It combines a **Model Context Protocol (MCP) tool server**, **LangGraph multi-agent pipelines**, and a **hybrid RAG search engine** to answer legislation questions and automatically assess loan applications against applicable banking law.

Built as a portfolio project to demonstrate production-grade patterns for agentic AI systems — not intended for live banking use without additional hardening.

> ⚠️ **This project is under active development.** Interfaces and the data model may still change.
>
> **In progress — per-article legal status propagation:** legal status is tracked per *article*


---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Notes on Production Readiness](#notes-on-production-readiness)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Legislation ingestion** — Upload PDF legislation files; the system parses, embeds, and indexes every article automatically.
- **Hybrid RAG search** — Combines semantic vector search, BM25 lexical ranking, and exact keyword matching with Reciprocal Rank Fusion (RRF) for best-of-all retrieval.
- **Legal advisor agent** — A LangGraph StateGraph that plans searches, traverses legislation relationship graphs, synthesizes cited answers, and self-critiques before responding.
- **Loan assessment agent** — A second LangGraph agent that loads a loan application, fetches the linked customer profile (credit score, payment history, existing debt), consults the legal agent for compliance questions, and produces a structured risk assessment saved to the database.
- **Customer profiles** — Full credit profiles with employment status, income, payment history, and credit score tracking.
- **Human-in-the-loop** — Both agents can pause mid-run and surface a clarification question to the user via the API; the conversation resumes when the user replies.
- **MCP tool server** — All data operations (legislation lookup, loan CRUD, customer retrieval) are exposed as MCP tools so agents can call them without direct DB access.
- **REST API** — FastAPI application with 16 endpoints covering ingestion, legal Q&A, loan management, and customer management.
- **303 tests** — Unit, integration, and HTTP-level tests across all layers.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI (REST)                        │
│  POST /ingest  POST /ask  /loans/*  /customers/*             │
└───────────────┬──────────────────────────┬───────────────────┘
                │                          │
    ┌───────────▼──────────┐   ┌───────────▼──────────┐
    │    Legal Agent       │   │    Loan Agent         │
    │  (LangGraph graph)   │   │  (LangGraph graph)    │
    │  plan → search →     │   │  load loan →          │
    │  traverse → critique │   │  fetch customer →     │
    └───────────┬──────────┘   │  plan → legal agent → │
                │              │  synthesize → save    │
                │              └───────────┬───────────┘
                │                          │
    ┌───────────▼──────────────────────────▼───────────┐
    │              MCP Tool Server (stdio)             │
    │  legal_lookup · loans · customers                │
    └─────────┬──────────────────┬────────────────────-┘
              │                  │
   ┌──────────▼────────┐  ┌──────▼─────────────┐
   │  ChromaDB +       │  │  PostgreSQL         │
   │  BM25 index       │  │  (Loan, Assessment, │
   │  (article embeds) │  │   Customer, Payment)│
   └───────────────────┘  └────────────────────-┘
```

### Agent flows

**Legal Agent**
```
plan_search → [ask_clarification?] → hybrid_search → retrieve_articles
           → traverse_parents → evaluate_relationships → [traverse_children?]
           → synthesize → critique → [retry synthesize?] → END
```

**Loan Agent**
```
load_loan → fetch_customer_context → plan_assessment
          → [ask_clarification?] → [consult_legal_agent?]
          → synthesize_assessment → save_result → END
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.115 + Uvicorn |
| Agent orchestration | LangGraph 0.2.70 |
| LLM | Anthropic Claude (via `langchain-anthropic`) |
| Tool protocol | Model Context Protocol — `fastmcp` + `langchain-mcp-adapters` |
| Vector search | ChromaDB 0.5 (cosine similarity) |
| Lexical search | BM25 (`rank-bm25`) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| PDF parsing | PyMuPDF + pdfplumber |
| Database | PostgreSQL (async via `asyncpg` + SQLAlchemy 2.0) |
| Validation | Pydantic v2 + pydantic-settings |
| Testing | pytest + pytest-asyncio + httpx |

---

## Project Structure

```
bank-legal-advisor/
├── src/
│   ├── agents/
│   │   ├── legal_agent.py        # LangGraph legal advisor
│   │   └── loan_agent.py         # LangGraph loan assessment
│   ├── api/
│   │   ├── main.py               # FastAPI app + lifespan
│   │   ├── deps.py               # Shared dependencies
│   │   ├── schemas.py            # API request/response models
│   │   └── routes/
│   │       ├── ingest.py         # POST /ingest
│   │       ├── legal.py          # POST /ask
│   │       ├── loans.py          # /loans/* CRUD + assess
│   │       └── customers.py      # /customers/* CRUD + payments
│   ├── core/
│   │   ├── config.py             # pydantic-settings (reads .env)
│   │   ├── llm.py                # Claude LLM wrapper
│   │   └── prompts.py            # All prompt templates
│   ├── db/
│   │   ├── models.py             # SQLAlchemy ORM models
│   │   ├── schemas.py            # Pydantic domain schemas & enums
│   │   ├── crud.py               # Async CRUD functions
│   │   └── session.py            # Engine + session factory
│   ├── mcp/
│   │   ├── app.py                # FastMCP instance
│   │   ├── server.py             # MCP server entry point
│   │   ├── client.py             # mcp_client() context manager
│   │   └── tools/
│   │       ├── legal_lookup.py   # Legislation search tools
│   │       ├── loans.py          # Loan read/write tools
│   │       └── customers.py      # Customer read-only tools
│   └── rag/
│       ├── pipeline.py           # End-to-end ingestion pipeline
│       ├── parser.py             # PDF → raw text
│       ├── ingestion.py          # LLM extraction (legislation + relationships)
│       ├── vectorstore.py        # ChromaDB + BM25 index + hybrid search
│       ├── embeddings.py         # SentenceTransformer (lazy-loaded)
│       ├── retriever.py          # High-level retrieval functions
│       ├── status_policy.py      # Pure article-status propagation rules
│       ├── reconcile.py          # Recompute article status on ingest (both directions)
│       └── graph.py              # Legislation relationship graph traversal
└── tests/
    ├── test_api.py               # FastAPI HTTP-level tests (39)
    ├── test_agent.py             # Legal agent unit tests (16)
    ├── test_loan_agent.py        # Loan agent unit tests (21)
    ├── test_rag.py               # RAG + vectorstore + retriever (71)
    ├── test_loans.py             # Loan CRUD + MCP tools (51)
    ├── test_customers.py         # Customer CRUD + MCP tools (35)
    ├── test_mcp_tools.py         # Legal MCP tools (19)
    └── test_ingestion.py / test_pipeline.py / test_parser.py
```

---

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (running locally or via Docker)
- An [Anthropic API key](https://console.anthropic.com/)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/mayarj/bank-legal-advisor.git
cd bank-legal-advisor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the database
createdb bank_legal_advisor       # or use your preferred PostgreSQL client

# 5. Copy the example environment file and fill in your values
cp .env.example .env
```

---

## Configuration

All configuration is read from `.env` (via `pydantic-settings`). Create this file from `.env.example`:

```env
# ── LLM ──────────────────────────────────────────────────────
ANTHROPIC_API_KEY=your-api-key-here
CLAUDE_MODEL=claude-sonnet-4-6
CLAUDE_TEMPERATURE=0.0
CLAUDE_MAX_TOKENS=8096

# ── Database ──────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/bank_legal_advisor

# ── Embeddings & Vector Store ─────────────────────────────────
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./data/chromadb
CHROMA_COLLECTION=legislation_articles

# ── Agent tuning ──────────────────────────────────────────────
MAX_CRITIQUE_RETRIES=2
GRAPH_K_DEPTH=2
SIMILARITY_N_RESULTS=5
```

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | *required* |
| `CLAUDE_MODEL` | Claude model ID | `claude-sonnet-4-6` |
| `DATABASE_URL` | Async PostgreSQL connection string | *required* |
| `EMBEDDING_MODEL` | SentenceTransformers model name | `all-MiniLM-L6-v2` |
| `CHROMA_PATH` | ChromaDB persistence directory | `./data/chromadb` |
| `MAX_CRITIQUE_RETRIES` | How many times the legal agent retries synthesis after a failed critique | `2` |
| `GRAPH_K_DEPTH` | Depth for legislation relationship traversal | `2` |
| `SIMILARITY_N_RESULTS` | Number of articles returned per search | `5` |

---

## Usage

### Start the server

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

On startup the server will:
1. Run `create_tables()` to bootstrap the database schema.
2. Start the MCP server as a subprocess and hold the connection open for the lifetime of the app.

---

### Ingest a legislation PDF

```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@/path/to/banking_law.pdf"
```

```json
{
  "message": "Legislation ingested successfully.",
  "code": "LAW-88-2003",
  "subject": "Banking loan collateral requirements",
  "status": "active",
  "articles_count": 12,
  "relationships_count": 3
}
```

---

### Ask a legal question

```bash
curl -X POST http://localhost:8000/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What collateral is required for a mortgage loan exceeding 500,000 EGP?"}'
```

```json
{
  "thread_id": "d3f1a2b4-...",
  "answer": "According to [LAW-88-2003 | Article 2 | active], all loans exceeding 50,000 units must be secured by real estate collateral. ..."
}
```

If the agent needs clarification, it returns:

```json
{
  "thread_id": "d3f1a2b4-...",
  "needs_clarification": true,
  "question": "Are you asking about commercial or residential mortgage loans?"
}
```

Resume by sending the `thread_id` and your answer:

```bash
curl -X POST http://localhost:8000/ask/ \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "thread_id": "d3f1a2b4-...", "clarification": "Residential mortgage loans."}'
```

---

### Create and assess a loan

```bash
# 1. Create a loan
curl -X POST http://localhost:8000/loans/ \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_name": "Ahmed Ali",
    "loan_type": "mortgage",
    "amount": "250000.00",
    "purpose": "Purchase residential property",
    "duration_months": 360,
    "interest_rate": "4.75",
    "collateral_type": "real_estate",
    "applicant_credit_score": 720
  }'

# 2. Run the assessment agent
curl -X POST http://localhost:8000/loans/{loan_id}/assess
```

```json
{
  "thread_id": "f8c2...",
  "final_assessment": "1. APPLICANT SUMMARY\n...\n4. RISK ASSESSMENT\nRisk Level: low\n...",
  "risk_level": "low",
  "cited_articles": ["LAW-88-2003 | Article 2"],
  "assessment_saved": true
}
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ingest/` | Upload PDF → ingest legislation |
| `POST` | `/ask/` | Ask a legal question (supports interrupt/resume) |
| `POST` | `/loans/` | Create a loan application |
| `GET` | `/loans/` | List loans (`?status=pending\|approved\|…`) |
| `GET` | `/loans/{id}` | Get a loan by ID |
| `PUT` | `/loans/{id}/status` | Update loan status |
| `DELETE` | `/loans/{id}` | Delete a loan |
| `POST` | `/loans/{id}/assess` | Run loan assessment agent (supports interrupt/resume) |
| `POST` | `/customers/` | Create a customer profile |
| `GET` | `/customers/` | List all customers |
| `GET` | `/customers/by-national-id/{nid}` | Look up customer by national ID |
| `GET` | `/customers/{id}` | Get customer profile |
| `PUT` | `/customers/{id}/credit-score` | Update credit score (300–850) |
| `GET` | `/customers/{id}/loans` | Get all loans linked to a customer |
| `GET` | `/customers/{id}/payments` | Get payment history |
| `POST` | `/customers/{id}/payments` | Record a payment |

Full interactive documentation is available at `/docs` (Swagger UI) and `/redoc` when the server is running.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite uses:
- **In-memory SQLite** for all database tests (no PostgreSQL required).
- **EphemeralClient** for ChromaDB vectorstore tests.
- **Mocked LLM calls** — no Anthropic API key needed to run tests.

```
303 passed in ~20s
```

Test files and what they cover:

| File | Layer tested | Tests |
|---|---|---|
| `test_api.py` | FastAPI HTTP routes | 39 |
| `test_agent.py` | Legal agent graph | 16 |
| `test_loan_agent.py` | Loan agent graph | 21 |
| `test_rag.py` | RAG, BM25, hybrid search | 71 |
| `test_loans.py` | Loan CRUD + MCP tools | 51 |
| `test_customers.py` | Customer CRUD + MCP tools | 35 |
| `test_mcp_tools.py` | Legal MCP tools | 19 |
| `test_reconcile.py` | Article status propagation | 21 |
| `test_ingestion.py` + others | Parser, pipeline, ingestion | 30 |

---

## Notes on Production Readiness

This project is a portfolio demonstration. Before deploying to production the following would be required:

- **Authentication** — API key or JWT middleware on all endpoints (no auth is currently in place).
- **Database migrations** — Replace `create_tables()` with Alembic versioned migrations.
- **Secrets management** — Move `ANTHROPIC_API_KEY` and `DATABASE_URL` to a vault (e.g., AWS Secrets Manager).
- **MCP transport** — Switch from `stdio` subprocess transport to a persistent `SSE` or `WebSocket` transport for better scalability.
- **Rate limiting & observability** — Add request rate limiting, structured logging, and tracing.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Make your changes and add tests.
4. Ensure all tests pass: `pytest tests/`.
5. Open a pull request with a clear description of what changed and why.

---


## Acknowledgments

- [Anthropic](https://anthropic.com) — Claude API and the Model Context Protocol specification.
- [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful agent orchestration framework.
- [LangChain](https://github.com/langchain-ai/langchain) — LLM tooling and MCP client adapters.
- [FastMCP](https://github.com/jlowin/fastmcp) — Ergonomic Python MCP server framework.
- [ChromaDB](https://github.com/chroma-core/chroma) — Open-source embedding database.
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — Local embedding model (`all-MiniLM-L6-v2`).
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 lexical ranking implementation.
