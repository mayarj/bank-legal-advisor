import json
import sys
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

# Stub fastmcp before any src.mcp imports (same pattern as test_mcp_tools.py)
_mock_mcp = MagicMock()
_mock_mcp.tool.return_value = lambda f: f
sys.modules.setdefault("fastmcp", MagicMock(FastMCP=MagicMock(return_value=_mock_mcp)))

from src.db.crud import (
    create_assessment,
    create_loan,
    delete_loan,
    get_assessments_for_loan,
    get_latest_assessment,
    get_loan,
    list_loans,
    update_loan_status,
)
from src.db.models import Loan, LoanAssessment
from src.db.schemas import (
    AssessmentCreate,
    CollateralType,
    LoanCreate,
    LoanStatus,
    LoanType,
    RiskLevel,
)
from src.mcp.tools.loans import (
    _format_assessment,
    _format_loan,
    create_loan_record,
    delete_loan_record,
    get_loan_assessments,
    get_loan_details,
    list_loan_records,
    save_assessment,
    update_loan_status as tool_update_loan_status,
)
from tests.conftest import make_relationship


# ── CRUD fixtures ─────────────────────────────────────────────────────────────

def _loan_data(**overrides) -> LoanCreate:
    defaults = dict(
        applicant_name="Ahmed Ali",
        loan_type=LoanType.MORTGAGE,
        amount=Decimal("250000.00"),
        purpose="Purchase residential property",
        collateral_type=CollateralType.REAL_ESTATE,
        duration_months=360,
        interest_rate=Decimal("4.75"),
        applicant_credit_score=720,
    )
    defaults.update(overrides)
    return LoanCreate(**defaults)


def _assessment_data(loan_id: uuid.UUID, **overrides) -> AssessmentCreate:
    defaults = dict(
        loan_id=loan_id,
        legal_question="What collateral is required for a 250,000 mortgage?",
        answer="Real estate collateral is required per LAW-88-2003 Article 2.",
        cited_articles=["LAW-88-2003 | Article 2"],
        risk_level=RiskLevel.LOW,
    )
    defaults.update(overrides)
    return AssessmentCreate(**defaults)


# ── create_loan ───────────────────────────────────────────────────────────────

class TestCreateLoan:

    async def test_returns_loan_with_pending_status(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        assert loan.status == LoanStatus.PENDING

    async def test_returns_loan_with_correct_fields(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        assert loan.applicant_name == "Ahmed Ali"
        assert loan.loan_type == LoanType.MORTGAGE
        assert loan.amount == Decimal("250000.00")
        assert loan.duration_months == 360
        assert loan.interest_rate == Decimal("4.75")
        assert loan.applicant_credit_score == 720

    async def test_assigns_uuid(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        assert isinstance(loan.id, uuid.UUID)

    async def test_persists_to_database(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        row = (await db_session.execute(select(Loan).where(Loan.id == loan.id))).scalars().first()
        assert row is not None
        assert row.applicant_name == "Ahmed Ali"

    async def test_optional_credit_score_can_be_none(self, db_session):
        loan = await create_loan(db_session, _loan_data(applicant_credit_score=None))
        assert loan.applicant_credit_score is None


# ── get_loan ──────────────────────────────────────────────────────────────────

class TestGetLoan:

    async def test_returns_loan_for_existing_id(self, db_session):
        created = await create_loan(db_session, _loan_data())
        fetched = await get_loan(db_session, created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_returns_none_for_missing_id(self, db_session):
        result = await get_loan(db_session, uuid.uuid4())
        assert result is None


# ── list_loans ────────────────────────────────────────────────────────────────

class TestListLoans:

    async def test_returns_all_loans_when_no_filter(self, db_session):
        await create_loan(db_session, _loan_data())
        await create_loan(db_session, _loan_data(applicant_name="Sara"))
        loans = await list_loans(db_session)
        assert len(loans) == 2

    async def test_filters_by_status(self, db_session):
        pending = await create_loan(db_session, _loan_data())
        under_review = await create_loan(db_session, _loan_data(applicant_name="Sara"))
        under_review.status = LoanStatus.UNDER_REVIEW
        await db_session.flush()

        result = await list_loans(db_session, status=LoanStatus.PENDING)
        assert len(result) == 1
        assert result[0].id == pending.id

    async def test_returns_empty_list_when_no_matches(self, db_session):
        result = await list_loans(db_session, status=LoanStatus.APPROVED)
        assert result == []

    async def test_respects_limit(self, db_session):
        for i in range(5):
            await create_loan(db_session, _loan_data(applicant_name=f"Applicant {i}"))
        result = await list_loans(db_session, limit=3)
        assert len(result) == 3


# ── update_loan_status ────────────────────────────────────────────────────────

class TestUpdateLoanStatus:

    async def test_updates_status(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        updated = await update_loan_status(db_session, loan.id, LoanStatus.APPROVED)
        assert updated.status == LoanStatus.APPROVED

    async def test_returns_none_for_missing_loan(self, db_session):
        result = await update_loan_status(db_session, uuid.uuid4(), LoanStatus.APPROVED)
        assert result is None

    async def test_persists_status_change(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        await update_loan_status(db_session, loan.id, LoanStatus.REJECTED)
        refetched = await get_loan(db_session, loan.id)
        assert refetched.status == LoanStatus.REJECTED


# ── delete_loan ───────────────────────────────────────────────────────────────

class TestDeleteLoan:

    async def test_returns_true_for_existing_loan(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        result = await delete_loan(db_session, loan.id)
        assert result is True

    async def test_loan_is_removed_from_database(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        await delete_loan(db_session, loan.id)
        fetched = await get_loan(db_session, loan.id)
        assert fetched is None

    async def test_returns_false_for_missing_loan(self, db_session):
        result = await delete_loan(db_session, uuid.uuid4())
        assert result is False


# ── create_assessment ─────────────────────────────────────────────────────────

class TestCreateAssessment:

    async def test_saves_assessment_linked_to_loan(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        assessment = await create_assessment(db_session, _assessment_data(loan.id))
        assert assessment.loan_id == loan.id

    async def test_serializes_cited_articles_as_json(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        assessment = await create_assessment(db_session, _assessment_data(loan.id))
        stored = json.loads(assessment.cited_articles)
        assert stored == ["LAW-88-2003 | Article 2"]

    async def test_saves_correct_risk_level(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        assessment = await create_assessment(db_session, _assessment_data(loan.id, risk_level=RiskLevel.HIGH))
        assert assessment.risk_level == RiskLevel.HIGH

    async def test_persists_to_database(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        assessment = await create_assessment(db_session, _assessment_data(loan.id))
        row = (
            await db_session.execute(
                select(LoanAssessment).where(LoanAssessment.id == assessment.id)
            )
        ).scalars().first()
        assert row is not None


# ── get_assessments_for_loan / get_latest_assessment ─────────────────────────

class TestGetAssessments:

    async def test_returns_all_assessments_for_loan(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        await create_assessment(db_session, _assessment_data(loan.id, risk_level=RiskLevel.LOW))
        await create_assessment(db_session, _assessment_data(loan.id, risk_level=RiskLevel.HIGH))
        results = await get_assessments_for_loan(db_session, loan.id)
        assert len(results) == 2

    async def test_returns_empty_list_for_loan_with_no_assessments(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        results = await get_assessments_for_loan(db_session, loan.id)
        assert results == []

    async def test_get_latest_returns_an_assessment_when_one_exists(self, db_session):
        # SQLite timestamps are second-resolution so ordering between two rapid
        # inserts is non-deterministic; test that the function returns something.
        loan = await create_loan(db_session, _loan_data())
        created = await create_assessment(db_session, _assessment_data(loan.id))
        latest = await get_latest_assessment(db_session, loan.id)
        assert latest is not None
        assert latest.id == created.id

    async def test_get_latest_returns_none_when_no_assessments(self, db_session):
        loan = await create_loan(db_session, _loan_data())
        result = await get_latest_assessment(db_session, loan.id)
        assert result is None


# ── Formatters ────────────────────────────────────────────────────────────────

class TestFormatLoan:

    def _make_loan(self) -> Loan:
        loan = Loan(
            id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
            applicant_name="Ahmed Ali",
            loan_type=LoanType.MORTGAGE,
            amount=Decimal("250000.00"),
            purpose="Buy a house",
            status=LoanStatus.PENDING,
            collateral_type=CollateralType.REAL_ESTATE,
            duration_months=360,
            interest_rate=Decimal("4.75"),
            applicant_credit_score=720,
            created_at=None,
        )
        return loan

    def test_contains_loan_id(self):
        assert "12345678" in _format_loan(self._make_loan())

    def test_contains_applicant_name(self):
        assert "Ahmed Ali" in _format_loan(self._make_loan())

    def test_contains_loan_type_and_amount(self):
        result = _format_loan(self._make_loan())
        assert "mortgage" in result
        assert "250000" in result

    def test_contains_status(self):
        assert "pending" in _format_loan(self._make_loan())


class TestFormatAssessment:

    def _make_assessment(self) -> LoanAssessment:
        return LoanAssessment(
            id=uuid.uuid4(),
            loan_id=uuid.uuid4(),
            legal_question="What is required?",
            answer="Real estate collateral required.",
            cited_articles=json.dumps(["LAW-88-2003 | Article 2"]),
            risk_level=RiskLevel.LOW,
            created_at=None,
        )

    def test_contains_risk_level(self):
        assert "low" in _format_assessment(self._make_assessment())

    def test_contains_cited_articles(self):
        assert "LAW-88-2003 | Article 2" in _format_assessment(self._make_assessment())

    def test_contains_answer(self):
        assert "Real estate collateral required." in _format_assessment(self._make_assessment())

    def test_handles_invalid_json_in_cited_articles(self):
        a = self._make_assessment()
        a.cited_articles = "not valid json"
        result = _format_assessment(a)
        assert "none" in result


# ── MCP Tools ─────────────────────────────────────────────────────────────────

class TestCreateLoanRecordTool:

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_creates_loan_and_returns_formatted_string(self, mock_factory):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        created_loan = Loan(
            id=uuid.uuid4(), applicant_name="Ahmed Ali",
            loan_type=LoanType.MORTGAGE, amount=Decimal("250000"),
            purpose="Buy house", status=LoanStatus.PENDING,
            collateral_type=CollateralType.REAL_ESTATE,
            duration_months=360, interest_rate=Decimal("4.75"),
            applicant_credit_score=720, created_at=None,
        )
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.add = MagicMock()

        with patch("src.mcp.tools.loans.create_loan", new=AsyncMock(return_value=created_loan)):
            result = await create_loan_record(
                applicant_name="Ahmed Ali",
                loan_type="mortgage",
                amount=250000,
                purpose="Buy house",
                duration_months=360,
                interest_rate=4.75,
                collateral_type="real_estate",
                applicant_credit_score=720,
            )

        assert "Loan created" in result
        assert "Ahmed Ali" in result

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_not_found_when_loan_missing(self, mock_factory):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.mcp.tools.loans.get_loan", new=AsyncMock(return_value=None)):
            result = await get_loan_details(str(uuid.uuid4()))

        assert "not found" in result.lower()


class TestSaveAssessmentTool:

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_saves_assessment_and_returns_confirmation(self, mock_factory):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        assessment = LoanAssessment(
            id=uuid.uuid4(), loan_id=uuid.uuid4(),
            legal_question="What is required?",
            answer="Collateral needed.",
            cited_articles=json.dumps(["LAW-88-2003 | Article 2"]),
            risk_level=RiskLevel.LOW, created_at=None,
        )

        with patch("src.mcp.tools.loans.create_assessment", new=AsyncMock(return_value=assessment)):
            result = await save_assessment(
                loan_id=str(uuid.uuid4()),
                legal_question="What is required?",
                answer="Collateral needed.",
                risk_level="low",
                cited_articles='["LAW-88-2003 | Article 2"]',
            )

        assert "Assessment saved" in result
        assert "low" in result

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_handles_malformed_cited_articles_gracefully(self, mock_factory):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        assessment = LoanAssessment(
            id=uuid.uuid4(), loan_id=uuid.uuid4(),
            legal_question="Q", answer="A",
            cited_articles=json.dumps([]),
            risk_level=RiskLevel.MEDIUM, created_at=None,
        )

        with patch("src.mcp.tools.loans.create_assessment", new=AsyncMock(return_value=assessment)):
            result = await save_assessment(
                loan_id=str(uuid.uuid4()),
                legal_question="Q", answer="A",
                risk_level="medium",
                cited_articles="this is not json",
            )

        assert "Assessment saved" in result


# ── Shared tool-test helper ───────────────────────────────────────────────────

def _stub_factory(mock_factory) -> AsyncMock:
    """Wire up the AsyncSessionFactory mock and return the session mock."""
    session = AsyncMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_loan_obj(**overrides) -> Loan:
    defaults = dict(
        id=uuid.uuid4(), applicant_name="Ahmed Ali",
        loan_type=LoanType.MORTGAGE, amount=Decimal("250000"),
        purpose="Buy house", status=LoanStatus.PENDING,
        collateral_type=CollateralType.REAL_ESTATE,
        duration_months=360, interest_rate=Decimal("4.75"),
        applicant_credit_score=720, created_at=None,
    )
    defaults.update(overrides)
    return Loan(**defaults)


def _make_assessment_obj(loan_id: uuid.UUID, **overrides) -> LoanAssessment:
    defaults = dict(
        id=uuid.uuid4(), loan_id=loan_id,
        legal_question="What collateral is required?",
        answer="Real estate collateral required per LAW-88-2003 Article 2.",
        cited_articles=json.dumps(["LAW-88-2003 | Article 2"]),
        risk_level=RiskLevel.LOW, created_at=None,
    )
    defaults.update(overrides)
    return LoanAssessment(**defaults)


# ── TestGetLoanDetailsTool ────────────────────────────────────────────────────

class TestGetLoanDetailsTool:

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_formatted_loan_when_found(self, mock_factory):
        _stub_factory(mock_factory)
        loan = _make_loan_obj()

        with (
            patch("src.mcp.tools.loans.get_loan", new=AsyncMock(return_value=loan)),
            patch("src.mcp.tools.loans.get_latest_assessment", new=AsyncMock(return_value=None)),
        ):
            result = await get_loan_details(str(loan.id))

        assert "Ahmed Ali" in result
        assert "mortgage" in result
        assert "Latest Assessment" not in result

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_includes_latest_assessment_when_present(self, mock_factory):
        _stub_factory(mock_factory)
        loan = _make_loan_obj()
        assessment = _make_assessment_obj(loan.id)

        with (
            patch("src.mcp.tools.loans.get_loan", new=AsyncMock(return_value=loan)),
            patch("src.mcp.tools.loans.get_latest_assessment", new=AsyncMock(return_value=assessment)),
        ):
            result = await get_loan_details(str(loan.id))

        assert "Latest Assessment" in result
        assert "low" in result
        assert "LAW-88-2003 | Article 2" in result

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_not_found_for_missing_loan(self, mock_factory):
        _stub_factory(mock_factory)
        missing_id = str(uuid.uuid4())

        with patch("src.mcp.tools.loans.get_loan", new=AsyncMock(return_value=None)):
            result = await get_loan_details(missing_id)

        assert "not found" in result.lower()
        assert missing_id in result


# ── TestListLoanRecordsTool ───────────────────────────────────────────────────

class TestListLoanRecordsTool:

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_formatted_loans(self, mock_factory):
        _stub_factory(mock_factory)
        loans = [_make_loan_obj(), _make_loan_obj(applicant_name="Sara")]

        with patch("src.mcp.tools.loans._list_loans", new=AsyncMock(return_value=loans)):
            result = await list_loan_records()

        assert "Ahmed Ali" in result
        assert "Sara" in result
        assert "---" in result  # separator between records

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_no_loans_message_when_empty(self, mock_factory):
        _stub_factory(mock_factory)

        with patch("src.mcp.tools.loans._list_loans", new=AsyncMock(return_value=[])):
            result = await list_loan_records()

        assert "No loans found" in result

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_passes_status_filter_to_crud(self, mock_factory):
        _stub_factory(mock_factory)
        mock_list = AsyncMock(return_value=[])

        with patch("src.mcp.tools.loans._list_loans", new=mock_list):
            await list_loan_records(status="approved")

        call_kwargs = mock_list.call_args
        assert call_kwargs.kwargs["status"] == LoanStatus.APPROVED

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_passes_none_status_when_not_provided(self, mock_factory):
        _stub_factory(mock_factory)
        mock_list = AsyncMock(return_value=[])

        with patch("src.mcp.tools.loans._list_loans", new=mock_list):
            await list_loan_records()

        call_kwargs = mock_list.call_args
        assert call_kwargs.kwargs["status"] is None


# ── TestUpdateLoanStatusTool ──────────────────────────────────────────────────

class TestUpdateLoanStatusTool:

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_updated_loan_string(self, mock_factory):
        _stub_factory(mock_factory)
        loan = _make_loan_obj(status=LoanStatus.APPROVED)

        with patch("src.mcp.tools.loans._update_loan_status", new=AsyncMock(return_value=loan)):
            result = await tool_update_loan_status(str(loan.id), "approved")

        assert "Status updated" in result
        assert "approved" in result

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_passes_correct_status_enum_to_crud(self, mock_factory):
        _stub_factory(mock_factory)
        mock_update = AsyncMock(return_value=None)

        with patch("src.mcp.tools.loans._update_loan_status", new=mock_update):
            await tool_update_loan_status(str(uuid.uuid4()), "rejected")

        _, status_arg = mock_update.call_args.args[1], mock_update.call_args.args[2]
        assert status_arg == LoanStatus.REJECTED

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_not_found_for_missing_loan(self, mock_factory):
        _stub_factory(mock_factory)
        missing_id = str(uuid.uuid4())

        with patch("src.mcp.tools.loans._update_loan_status", new=AsyncMock(return_value=None)):
            result = await tool_update_loan_status(missing_id, "approved")

        assert "not found" in result.lower()


# ── TestGetLoanAssessmentsTool ────────────────────────────────────────────────

class TestGetLoanAssessmentsTool:

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_formatted_assessments(self, mock_factory):
        _stub_factory(mock_factory)
        loan_id = uuid.uuid4()
        assessments = [
            _make_assessment_obj(loan_id, risk_level=RiskLevel.HIGH),
            _make_assessment_obj(loan_id, risk_level=RiskLevel.LOW),
        ]

        with patch("src.mcp.tools.loans.get_assessments_for_loan", new=AsyncMock(return_value=assessments)):
            result = await get_loan_assessments(str(loan_id))

        assert "high" in result
        assert "low" in result
        assert "---" in result

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_no_assessments_message_when_empty(self, mock_factory):
        _stub_factory(mock_factory)
        loan_id = str(uuid.uuid4())

        with patch("src.mcp.tools.loans.get_assessments_for_loan", new=AsyncMock(return_value=[])):
            result = await get_loan_assessments(loan_id)

        assert "No assessments found" in result
        assert loan_id in result


class TestDeleteLoanRecordTool:

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_deleted_confirmation(self, mock_factory):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)
        loan_id = str(uuid.uuid4())

        with patch("src.mcp.tools.loans._delete_loan", new=AsyncMock(return_value=True)):
            result = await delete_loan_record(loan_id)

        assert "deleted" in result.lower()

    @patch("src.mcp.tools.loans.AsyncSessionFactory")
    async def test_returns_not_found_when_missing(self, mock_factory):
        mock_session = AsyncMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.mcp.tools.loans._delete_loan", new=AsyncMock(return_value=False)):
            result = await delete_loan_record(str(uuid.uuid4()))

        assert "not found" in result.lower()