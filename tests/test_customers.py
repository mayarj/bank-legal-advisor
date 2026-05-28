"""
Checkpoint: customer-profile — CustomerProfile + Payment models, CRUD, and read-only MCP tools.
"""
import sys
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub fastmcp before any src.mcp imports
_mock_mcp = MagicMock()
_mock_mcp.tool.return_value = lambda f: f
sys.modules.setdefault("fastmcp", MagicMock(FastMCP=MagicMock(return_value=_mock_mcp)))

from src.db.crud import (
    create_customer_profile,
    create_loan,
    create_payment,
    get_customer_by_national_id,
    get_customer_profile,
    get_loans_for_customer,
    get_payments_for_customer,
    get_payments_for_loan,
    list_customer_profiles,
    update_customer_credit_score,
)
from src.db.models import CustomerProfile, Payment
from src.db.schemas import (
    CollateralType,
    CustomerProfileCreate,
    EmploymentStatus,
    LoanCreate,
    LoanType,
    PaymentCreate,
    PaymentStatus,
)
from src.mcp.tools.customers import (
    _format_loan,
    _format_payment,
    _format_profile,
    get_customer_loans,
    get_customer_payment_history,
    get_customer_profile_by_id,
    get_customer_profile_by_national_id,
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _profile_data(**overrides) -> CustomerProfileCreate:
    defaults = dict(
        national_id="EG-123456789",
        full_name="Fatima Hassan",
        date_of_birth=date(1990, 5, 15),
        phone="+20-100-0000000",
        email="fatima@example.com",
        employment_status=EmploymentStatus.EMPLOYED,
        monthly_income=Decimal("12000.00"),
        existing_debt_amount=Decimal("5000.00"),
        credit_score=710,
    )
    defaults.update(overrides)
    return CustomerProfileCreate(**defaults)


def _payment_data(customer_id: uuid.UUID, loan_id=None, **overrides) -> PaymentCreate:
    defaults = dict(
        customer_id=customer_id,
        loan_id=loan_id,
        amount=Decimal("1500.00"),
        due_date=date(2024, 1, 1),
        paid_date=date(2024, 1, 1),
        status=PaymentStatus.ON_TIME,
    )
    defaults.update(overrides)
    return PaymentCreate(**defaults)


def _loan_data(customer_id=None, **overrides) -> LoanCreate:
    defaults = dict(
        applicant_name="Fatima Hassan",
        loan_type=LoanType.MORTGAGE,
        amount=Decimal("200000.00"),
        purpose="Buy apartment",
        duration_months=240,
        interest_rate=Decimal("5.00"),
        customer_id=customer_id,
    )
    defaults.update(overrides)
    return LoanCreate(**defaults)


# ── CustomerProfile CRUD ──────────────────────────────────────────────────────

class TestCreateCustomerProfile:

    async def test_creates_profile_with_all_fields(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())

        assert profile.id is not None
        assert profile.national_id == "EG-123456789"
        assert profile.full_name == "Fatima Hassan"
        assert profile.employment_status == EmploymentStatus.EMPLOYED
        assert profile.credit_score == 710

    async def test_credit_score_updated_at_set_when_score_provided(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data(credit_score=720))

        assert profile.credit_score_updated_at is not None

    async def test_credit_score_updated_at_is_none_when_no_score(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data(credit_score=None))

        assert profile.credit_score_updated_at is None

    async def test_defaults_existing_debt_to_zero(self, db_session):
        profile = await create_customer_profile(
            db_session, _profile_data(existing_debt_amount=Decimal("0.00"))
        )

        assert profile.existing_debt_amount == Decimal("0.00")


class TestGetCustomerProfile:

    async def test_returns_profile_by_id(self, db_session):
        created = await create_customer_profile(db_session, _profile_data())

        fetched = await get_customer_profile(db_session, created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.full_name == "Fatima Hassan"

    async def test_returns_none_for_missing_id(self, db_session):
        result = await get_customer_profile(db_session, uuid.uuid4())

        assert result is None

    async def test_returns_profile_by_national_id(self, db_session):
        await create_customer_profile(db_session, _profile_data())

        result = await get_customer_by_national_id(db_session, "EG-123456789")

        assert result is not None
        assert result.national_id == "EG-123456789"

    async def test_returns_none_for_unknown_national_id(self, db_session):
        result = await get_customer_by_national_id(db_session, "UNKNOWN-999")

        assert result is None

    async def test_list_returns_all_profiles(self, db_session):
        await create_customer_profile(db_session, _profile_data(national_id="EG-001"))
        await create_customer_profile(db_session, _profile_data(national_id="EG-002"))

        profiles = await list_customer_profiles(db_session)

        assert len(profiles) == 2


class TestUpdateCustomerCreditScore:

    async def test_updates_score_and_timestamp(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data(credit_score=650))

        updated = await update_customer_credit_score(db_session, profile.id, 720)

        assert updated is not None
        assert updated.credit_score == 720
        assert updated.credit_score_updated_at is not None

    async def test_returns_none_for_missing_customer(self, db_session):
        result = await update_customer_credit_score(db_session, uuid.uuid4(), 700)

        assert result is None


# ── Payment CRUD ──────────────────────────────────────────────────────────────

class TestCreatePayment:

    async def test_creates_payment_linked_to_customer(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())

        payment = await create_payment(db_session, _payment_data(profile.id))

        assert payment.id is not None
        assert payment.customer_id == profile.id
        assert payment.status == PaymentStatus.ON_TIME

    async def test_creates_payment_without_loan_ref(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())

        payment = await create_payment(db_session, _payment_data(profile.id, loan_id=None))

        assert payment.loan_id is None

    async def test_creates_payment_with_loan_ref(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())
        loan = await create_loan(db_session, _loan_data(customer_id=profile.id))

        payment = await create_payment(db_session, _payment_data(profile.id, loan_id=loan.id))

        assert payment.loan_id == loan.id


class TestGetPayments:

    async def test_get_payments_for_customer_returns_all(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())
        await create_payment(db_session, _payment_data(profile.id, status=PaymentStatus.ON_TIME))
        await create_payment(db_session, _payment_data(profile.id, status=PaymentStatus.LATE))

        payments = await get_payments_for_customer(db_session, profile.id)

        assert len(payments) == 2

    async def test_get_payments_for_customer_returns_empty_for_unknown(self, db_session):
        payments = await get_payments_for_customer(db_session, uuid.uuid4())

        assert payments == []

    async def test_get_payments_for_loan_returns_linked_payments(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())
        loan = await create_loan(db_session, _loan_data(customer_id=profile.id))
        await create_payment(db_session, _payment_data(profile.id, loan_id=loan.id))
        await create_payment(db_session, _payment_data(profile.id, loan_id=None))

        loan_payments = await get_payments_for_loan(db_session, loan.id)

        assert len(loan_payments) == 1
        assert loan_payments[0].loan_id == loan.id


# ── Customer-Loan link ────────────────────────────────────────────────────────

class TestGetLoansForCustomer:

    async def test_returns_loans_linked_to_customer(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())
        await create_loan(db_session, _loan_data(customer_id=profile.id))
        await create_loan(db_session, _loan_data(customer_id=profile.id))

        loans = await get_loans_for_customer(db_session, profile.id)

        assert len(loans) == 2
        assert all(loan.customer_id == profile.id for loan in loans)

    async def test_returns_empty_for_customer_with_no_loans(self, db_session):
        profile = await create_customer_profile(db_session, _profile_data())

        loans = await get_loans_for_customer(db_session, profile.id)

        assert loans == []

    async def test_does_not_return_loans_from_other_customers(self, db_session):
        profile1 = await create_customer_profile(db_session, _profile_data(national_id="EG-001"))
        profile2 = await create_customer_profile(db_session, _profile_data(national_id="EG-002"))
        await create_loan(db_session, _loan_data(customer_id=profile1.id))

        loans = await get_loans_for_customer(db_session, profile2.id)

        assert loans == []


# ── Formatters ────────────────────────────────────────────────────────────────

class TestFormatters:

    def _make_profile(self, **overrides):
        p = MagicMock(spec=CustomerProfile)
        p.id = uuid.uuid4()
        p.national_id = "EG-123"
        p.full_name = "Fatima Hassan"
        p.date_of_birth = date(1990, 5, 15)
        p.phone = "+20-100-0000"
        p.email = "f@test.com"
        p.employment_status = MagicMock(value="employed")
        p.monthly_income = Decimal("10000")
        p.existing_debt_amount = Decimal("2000")
        p.credit_score = 700
        p.credit_score_updated_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for k, v in overrides.items():
            setattr(p, k, v)
        return p

    def test_format_profile_contains_name(self):
        assert "Fatima Hassan" in _format_profile(self._make_profile())

    def test_format_profile_contains_credit_score(self):
        assert "700" in _format_profile(self._make_profile())

    def test_format_profile_shows_dti_when_income_and_debt_present(self):
        result = _format_profile(self._make_profile())
        assert "Debt-to-Income" in result

    def test_format_profile_no_dti_when_no_income(self):
        result = _format_profile(self._make_profile(monthly_income=None))
        assert "Debt-to-Income" not in result

    def test_format_profile_shows_na_for_missing_credit_score(self):
        result = _format_profile(self._make_profile(credit_score=None, credit_score_updated_at=None))
        assert "N/A" in result

    def test_format_payment_shows_status(self):
        p = MagicMock(spec=Payment)
        p.status = MagicMock(value="late")
        p.amount = Decimal("1000")
        p.due_date = date(2024, 3, 1)
        p.paid_date = None
        p.loan_id = None
        p.notes = None
        assert "LATE" in _format_payment(p)

    def test_format_payment_shows_paid_date(self):
        p = MagicMock(spec=Payment)
        p.status = MagicMock(value="on_time")
        p.amount = Decimal("500")
        p.due_date = date(2024, 3, 1)
        p.paid_date = date(2024, 2, 28)
        p.loan_id = None
        p.notes = None
        assert "2024-02-28" in _format_payment(p)


# ── MCP tools ─────────────────────────────────────────────────────────────────

def _stub_factory(session_mock):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session_mock)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_profile_obj(national_id="EG-001", credit_score=700):
    p = MagicMock(spec=CustomerProfile)
    p.id = uuid.uuid4()
    p.national_id = national_id
    p.full_name = "Test Customer"
    p.date_of_birth = date(1985, 1, 1)
    p.phone = None
    p.email = None
    p.employment_status = MagicMock(value="employed")
    p.monthly_income = Decimal("8000")
    p.existing_debt_amount = Decimal("0")
    p.credit_score = credit_score
    p.credit_score_updated_at = None
    return p


def _make_payment_obj(customer_id, status="on_time"):
    p = MagicMock(spec=Payment)
    p.id = uuid.uuid4()
    p.customer_id = customer_id
    p.loan_id = None
    p.amount = Decimal("500")
    p.due_date = date(2024, 1, 1)
    p.paid_date = date(2024, 1, 1)
    p.status = MagicMock(value=status)
    p.notes = None
    return p


class TestGetCustomerProfileById:

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers._get_customer_profile", new_callable=AsyncMock)
    async def test_returns_formatted_profile(self, mock_get, mock_factory):
        profile = _make_profile_obj()
        mock_get.return_value = profile
        mock_factory.return_value = _stub_factory(MagicMock())

        result = await get_customer_profile_by_id(str(profile.id))

        assert "Test Customer" in result

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers._get_customer_profile", new_callable=AsyncMock)
    async def test_returns_not_found_for_missing_id(self, mock_get, mock_factory):
        mock_get.return_value = None
        mock_factory.return_value = _stub_factory(MagicMock())
        cid = str(uuid.uuid4())

        result = await get_customer_profile_by_id(cid)

        assert "not found" in result.lower()


class TestGetCustomerProfileByNationalId:

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers._get_customer_by_national_id", new_callable=AsyncMock)
    async def test_returns_formatted_profile(self, mock_get, mock_factory):
        profile = _make_profile_obj(national_id="EG-999")
        mock_get.return_value = profile
        mock_factory.return_value = _stub_factory(MagicMock())

        result = await get_customer_profile_by_national_id("EG-999")

        assert "Test Customer" in result

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers._get_customer_by_national_id", new_callable=AsyncMock)
    async def test_returns_not_found_for_unknown_national_id(self, mock_get, mock_factory):
        mock_get.return_value = None
        mock_factory.return_value = _stub_factory(MagicMock())

        result = await get_customer_profile_by_national_id("UNKNOWN")

        assert "No customer found" in result


class TestGetCustomerPaymentHistory:

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers.get_payments_for_customer", new_callable=AsyncMock)
    async def test_returns_summary_and_records(self, mock_get, mock_factory):
        cid = uuid.uuid4()
        mock_get.return_value = [
            _make_payment_obj(cid, "on_time"),
            _make_payment_obj(cid, "late"),
        ]
        mock_factory.return_value = _stub_factory(MagicMock())

        result = await get_customer_payment_history(str(cid))

        assert "On-time: 1" in result
        assert "Late: 1" in result
        assert "Total: 2" in result

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers.get_payments_for_customer", new_callable=AsyncMock)
    async def test_returns_no_records_message(self, mock_get, mock_factory):
        mock_get.return_value = []
        mock_factory.return_value = _stub_factory(MagicMock())

        result = await get_customer_payment_history(str(uuid.uuid4()))

        assert "No payment records" in result


class TestGetCustomerLoans:

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers.get_loans_for_customer", new_callable=AsyncMock)
    async def test_returns_formatted_loans(self, mock_get, mock_factory):
        from src.db.models import Loan
        from src.db.schemas import LoanStatus
        loan = MagicMock(spec=Loan)
        loan.id = uuid.uuid4()
        loan.loan_type = MagicMock(value="mortgage")
        loan.amount = Decimal("200000")
        loan.status = MagicMock(value="pending")
        loan.duration_months = 240
        loan.interest_rate = Decimal("5.0")
        loan.collateral_type = None
        loan.purpose = "Buy apartment"
        loan.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        mock_get.return_value = [loan]
        mock_factory.return_value = _stub_factory(MagicMock())

        result = await get_customer_loans(str(uuid.uuid4()))

        assert "mortgage" in result
        assert "200000" in result

    @patch("src.mcp.tools.customers.AsyncSessionFactory")
    @patch("src.mcp.tools.customers.get_loans_for_customer", new_callable=AsyncMock)
    async def test_returns_no_loans_message(self, mock_get, mock_factory):
        mock_get.return_value = []
        mock_factory.return_value = _stub_factory(MagicMock())

        result = await get_customer_loans(str(uuid.uuid4()))

        assert "No loans found" in result