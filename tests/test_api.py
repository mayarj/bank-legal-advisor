"""
Checkpoint: API layer — FastAPI route tests for all 16 endpoints.
Uses in-memory SQLite so CRUD runs against a real schema, and mocks
agents + pipeline so no LLM or MCP server is needed.
"""
import sys
import uuid
from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Stub fastmcp before any src.mcp import (same pattern as other test files)
_mock_mcp = MagicMock()
_mock_mcp.tool.return_value = lambda f: f
sys.modules.setdefault("fastmcp", MagicMock(FastMCP=MagicMock(return_value=_mock_mcp)))

from src.api.deps import get_db, get_tools
from src.api.main import app
from src.db.base import Base
from src.db.crud import create_customer_profile, create_loan
from src.db.schemas import (
    CollateralType,
    CustomerProfileCreate,
    EmploymentStatus,
    LoanCreate,
    LoanType,
    PaymentStatus,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def test_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def client(test_engine):
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db]    = _override_get_db
    app.dependency_overrides[get_tools] = lambda: []

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Shared test-data helpers ───────────────────────────────────────────────────

def _loan_payload(**overrides):
    base = {
        "applicant_name": "Ahmed Ali",
        "loan_type": "mortgage",
        "amount": "250000.00",
        "purpose": "Buy apartment",
        "duration_months": 240,
        "interest_rate": "5.00",
    }
    base.update(overrides)
    return base


def _customer_payload(**overrides):
    base = {
        "national_id": f"EG-{uuid.uuid4().hex[:6]}",
        "full_name": "Fatima Hassan",
        "employment_status": "employed",
        "existing_debt_amount": "0.00",
    }
    base.update(overrides)
    return base


async def _seed_loan(test_engine) -> str:
    """Create a loan in the test DB and return its string ID."""
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        loan = await create_loan(session, LoanCreate(
            applicant_name="Ahmed Ali",
            loan_type=LoanType.MORTGAGE,
            amount=Decimal("200000"),
            purpose="Buy apartment",
            duration_months=240,
            interest_rate=Decimal("5.00"),
        ))
        await session.commit()
        return str(loan.id)


async def _seed_customer(test_engine) -> str:
    """Create a customer profile in the test DB and return its string ID."""
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        profile = await create_customer_profile(session, CustomerProfileCreate(
            national_id=f"EG-{uuid.uuid4().hex[:6]}",
            full_name="Fatima Hassan",
            employment_status=EmploymentStatus.EMPLOYED,
        ))
        await session.commit()
        return str(profile.id)


# ── Interrupt helpers ─────────────────────────────────────────────────────────

def _make_interrupted_agent(interrupt_message: str):
    """Mock agent that returns no final answer (graph paused at interrupt)."""
    class _Interrupt:
        value = {"message": interrupt_message}

    class _Task:
        interrupts = [_Interrupt()]

    class _Snapshot:
        tasks = [_Task()]

    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value={"messages": []})
    mock.aget_state = AsyncMock(return_value=_Snapshot())
    return mock


def _make_legal_agent(answer: str = "According to [LAW-88-2003 | Article 2 | active], loans must be secured."):
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value={"draft_answer": answer, "critique_passed": True})
    return mock


def _make_loan_assessment_agent(risk: str = "medium"):
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value={
        "assessment_saved": True,
        "final_assessment": "Loan assessment complete.",
        "risk_level": risk,
        "cited_articles": ["LAW-88-2003 | Article 2"],
    })
    return mock


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:

    async def test_returns_200_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ── Ingest ────────────────────────────────────────────────────────────────────

class TestIngest:

    @patch("src.api.routes.ingest.run_pipeline")
    async def test_pdf_ingested_successfully(self, mock_pipeline, client):
        leg = MagicMock()
        leg.code = "LAW-88-2003"
        leg.subject = "Banking collateral"
        leg.status = MagicMock(value="active")
        leg.articles = {"1": "Article text", "2": "More text"}
        leg.relationships = []
        mock_pipeline.return_value = leg

        response = await client.post(
            "/ingest/",
            files={"file": ("law.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["code"] == "LAW-88-2003"
        assert body["articles_count"] == 2

    @patch("src.api.routes.ingest.run_pipeline_from_text")
    async def test_txt_ingested_successfully(self, mock_pipeline, client):
        leg = MagicMock()
        leg.code = "LAW-88-2003"
        leg.subject = "Banking collateral"
        leg.status = MagicMock(value="active")
        leg.articles = {"1": "Article text"}
        leg.relationships = []
        mock_pipeline.return_value = leg

        response = await client.post(
            "/ingest/",
            files={"file": ("law.txt", b"Article 1: text", "text/plain")},
        )

        assert response.status_code == 200
        assert response.json()["code"] == "LAW-88-2003"
        mock_pipeline.assert_awaited_once()

    async def test_unsupported_type_rejected_with_400(self, client):
        response = await client.post(
            "/ingest/",
            files={"file": ("law.docx", b"some text", "application/octet-stream")},
        )
        assert response.status_code == 400

    @patch("src.api.routes.ingest.run_pipeline")
    async def test_unextractable_pdf_returns_422(self, mock_pipeline, client):
        mock_pipeline.return_value = None

        response = await client.post(
            "/ingest/",
            files={"file": ("empty.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert response.status_code == 422


# ── Legal advisor ─────────────────────────────────────────────────────────────

class TestAsk:

    @patch("src.api.routes.legal.build_legal_agent")
    async def test_returns_answer(self, mock_build, client):
        mock_build.return_value = _make_legal_agent()

        response = await client.post("/ask/", json={"question": "What are collateral rules?"})

        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "thread_id" in body

    @patch("src.api.routes.legal.build_legal_agent")
    async def test_returns_clarification_when_interrupted(self, mock_build, client):
        mock_build.return_value = _make_interrupted_agent("Which article are you asking about?")

        response = await client.post("/ask/", json={"question": "Tell me about the law"})

        assert response.status_code == 200
        body = response.json()
        assert body["needs_clarification"] is True
        assert "thread_id" in body
        assert "Which article" in body["question"]

    @patch("src.api.routes.legal.build_legal_agent")
    async def test_resumes_with_clarification(self, mock_build, client):
        mock_build.return_value = _make_legal_agent("After clarification: LAW-88-2003 applies.")

        tid = str(uuid.uuid4())
        response = await client.post(
            "/ask/",
            json={"question": "What about loans?", "thread_id": tid, "clarification": "Mortgage loans only"},
        )

        assert response.status_code == 200
        assert response.json()["answer"] != ""

    @patch("src.api.routes.legal.build_legal_agent")
    async def test_generates_thread_id_when_none_provided(self, mock_build, client):
        mock_build.return_value = _make_legal_agent()

        r1 = await client.post("/ask/", json={"question": "Q1"})
        r2 = await client.post("/ask/", json={"question": "Q2"})

        assert r1.json()["thread_id"] != r2.json()["thread_id"]


# ── Loans CRUD ────────────────────────────────────────────────────────────────

class TestLoansCRUD:

    async def test_create_loan_returns_201(self, client):
        response = await client.post("/loans/", json=_loan_payload())
        assert response.status_code == 201
        body = response.json()
        assert body["applicant_name"] == "Ahmed Ali"
        assert body["status"] == "pending"
        assert "id" in body

    async def test_list_loans_returns_200(self, client):
        await client.post("/loans/", json=_loan_payload())
        response = await client.get("/loans/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_list_loans_filters_by_status(self, client):
        await client.post("/loans/", json=_loan_payload())
        response = await client.get("/loans/?status=pending")
        assert response.status_code == 200
        assert all(loan["status"] == "pending" for loan in response.json())

    async def test_list_loans_invalid_status_returns_400(self, client):
        response = await client.get("/loans/?status=invalid_status")
        assert response.status_code == 400

    async def test_get_existing_loan_returns_200(self, client, test_engine):
        lid = await _seed_loan(test_engine)
        response = await client.get(f"/loans/{lid}")
        assert response.status_code == 200
        assert response.json()["id"] == lid

    async def test_get_missing_loan_returns_404(self, client):
        response = await client.get(f"/loans/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_update_loan_status(self, client, test_engine):
        lid = await _seed_loan(test_engine)
        response = await client.put(f"/loans/{lid}/status", json={"status": "approved"})
        assert response.status_code == 200
        assert response.json()["status"] == "approved"

    async def test_update_status_invalid_value_returns_400(self, client, test_engine):
        lid = await _seed_loan(test_engine)
        response = await client.put(f"/loans/{lid}/status", json={"status": "flying"})
        assert response.status_code == 400

    async def test_update_status_missing_loan_returns_404(self, client):
        response = await client.put(f"/loans/{uuid.uuid4()}/status", json={"status": "approved"})
        assert response.status_code == 404

    async def test_delete_loan_returns_200(self, client, test_engine):
        lid = await _seed_loan(test_engine)
        response = await client.delete(f"/loans/{lid}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    async def test_delete_missing_loan_returns_404(self, client):
        response = await client.delete(f"/loans/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_deleted_loan_no_longer_found(self, client, test_engine):
        lid = await _seed_loan(test_engine)
        await client.delete(f"/loans/{lid}")
        response = await client.get(f"/loans/{lid}")
        assert response.status_code == 404


# ── Loan assessment ───────────────────────────────────────────────────────────

class TestLoanAssess:

    @patch("src.api.routes.loans.build_loan_agent")
    async def test_assess_returns_final_assessment(self, mock_build, client, test_engine):
        mock_build.return_value = _make_loan_assessment_agent()
        lid = await _seed_loan(test_engine)

        response = await client.post(f"/loans/{lid}/assess")

        assert response.status_code == 200
        body = response.json()
        assert body["assessment_saved"] is True
        assert body["risk_level"] == "medium"
        assert "thread_id" in body

    @patch("src.api.routes.loans.build_loan_agent")
    async def test_assess_returns_clarification_when_interrupted(self, mock_build, client, test_engine):
        mock_build.return_value = _make_interrupted_agent("Please confirm the applicant's net income.")
        lid = await _seed_loan(test_engine)

        response = await client.post(f"/loans/{lid}/assess")

        assert response.status_code == 200
        body = response.json()
        assert body["needs_clarification"] is True
        assert "net income" in body["question"]

    @patch("src.api.routes.loans.build_loan_agent")
    async def test_assess_resumes_with_clarification(self, mock_build, client, test_engine):
        mock_build.return_value = _make_loan_assessment_agent()
        lid = await _seed_loan(test_engine)
        tid = str(uuid.uuid4())

        response = await client.post(
            f"/loans/{lid}/assess",
            json={"thread_id": tid, "clarification": "Monthly income is 15000 EGP"},
        )

        assert response.status_code == 200
        assert response.json()["assessment_saved"] is True

    async def test_assess_missing_loan_returns_404(self, client):
        response = await client.post(f"/loans/{uuid.uuid4()}/assess")
        assert response.status_code == 404


# ── Customers CRUD ────────────────────────────────────────────────────────────

class TestCustomersCRUD:

    async def test_create_customer_returns_201(self, client):
        response = await client.post("/customers/", json=_customer_payload())
        assert response.status_code == 201
        body = response.json()
        assert body["full_name"] == "Fatima Hassan"
        assert "id" in body

    async def test_list_customers_returns_200(self, client):
        await client.post("/customers/", json=_customer_payload())
        response = await client.get("/customers/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) >= 1

    async def test_get_customer_by_id(self, client, test_engine):
        cid = await _seed_customer(test_engine)
        response = await client.get(f"/customers/{cid}")
        assert response.status_code == 200
        assert response.json()["id"] == cid

    async def test_get_missing_customer_returns_404(self, client):
        response = await client.get(f"/customers/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_get_customer_by_national_id(self, client):
        nid = f"EG-{uuid.uuid4().hex[:6]}"
        await client.post("/customers/", json=_customer_payload(national_id=nid))

        response = await client.get(f"/customers/by-national-id/{nid}")
        assert response.status_code == 200
        assert response.json()["national_id"] == nid

    async def test_get_unknown_national_id_returns_404(self, client):
        response = await client.get("/customers/by-national-id/UNKNOWN-999")
        assert response.status_code == 404

    async def test_update_credit_score(self, client, test_engine):
        cid = await _seed_customer(test_engine)
        response = await client.put(f"/customers/{cid}/credit-score", json={"credit_score": 750})
        assert response.status_code == 200
        assert response.json()["credit_score"] == 750

    async def test_credit_score_out_of_range_returns_400(self, client, test_engine):
        cid = await _seed_customer(test_engine)
        response = await client.put(f"/customers/{cid}/credit-score", json={"credit_score": 900})
        assert response.status_code == 400

    async def test_credit_score_too_low_returns_400(self, client, test_engine):
        cid = await _seed_customer(test_engine)
        response = await client.put(f"/customers/{cid}/credit-score", json={"credit_score": 100})
        assert response.status_code == 400

    async def test_update_credit_score_missing_customer_returns_404(self, client):
        response = await client.put(f"/customers/{uuid.uuid4()}/credit-score", json={"credit_score": 700})
        assert response.status_code == 404

    async def test_get_customer_loans_empty(self, client, test_engine):
        cid = await _seed_customer(test_engine)
        response = await client.get(f"/customers/{cid}/loans")
        assert response.status_code == 200
        assert response.json() == []

    async def test_get_customer_loans_missing_customer_returns_404(self, client):
        response = await client.get(f"/customers/{uuid.uuid4()}/loans")
        assert response.status_code == 404

    async def test_add_payment_to_customer(self, client, test_engine):
        cid = await _seed_customer(test_engine)
        response = await client.post(
            f"/customers/{cid}/payments",
            json={
                "amount": "500.00",
                "due_date": "2024-01-01",
                "paid_date": "2024-01-01",
                "status": "on_time",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["customer_id"] == cid
        assert body["status"] == "on_time"

    async def test_get_customer_payments(self, client, test_engine):
        cid = await _seed_customer(test_engine)
        await client.post(
            f"/customers/{cid}/payments",
            json={"amount": "200.00", "due_date": "2024-02-01", "status": "late"},
        )
        response = await client.get(f"/customers/{cid}/payments")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["status"] == "late"

    async def test_get_payments_missing_customer_returns_404(self, client):
        response = await client.get(f"/customers/{uuid.uuid4()}/payments")
        assert response.status_code == 404