from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.db.schemas import (
    CollateralType,
    EmploymentStatus,
    LoanStatus,
    LoanType,
    PaymentStatus,
    RiskLevel,
)


# ── ORM response models ────────────────────────────────────────────────────────

class LoanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    applicant_name: str
    loan_type: LoanType
    amount: Decimal
    purpose: str
    status: LoanStatus
    collateral_type: CollateralType | None
    duration_months: int
    interest_rate: Decimal
    applicant_credit_score: int | None
    customer_id: UUID | None
    created_at: datetime


class AssessmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    loan_id: UUID
    legal_question: str
    answer: str
    cited_articles: str   # JSON string stored as-is; callers parse if needed
    risk_level: RiskLevel
    created_at: datetime


class CustomerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    national_id: str
    full_name: str
    date_of_birth: date | None
    phone: str | None
    email: str | None
    employment_status: EmploymentStatus
    monthly_income: Decimal | None
    existing_debt_amount: Decimal
    credit_score: int | None
    credit_score_updated_at: datetime | None
    created_at: datetime


class PaymentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    customer_id: UUID
    loan_id: UUID | None
    amount: Decimal
    due_date: date
    paid_date: date | None
    status: PaymentStatus
    notes: str | None
    created_at: datetime


# ── Agent request/response models ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    thread_id: str | None = None
    clarification: str | None = None  # non-null resumes a paused agent


class AskResponse(BaseModel):
    thread_id: str
    answer: str


class ClarificationRequired(BaseModel):
    thread_id: str
    needs_clarification: bool = True
    question: str
    options: list = []


class AssessLoanRequest(BaseModel):
    thread_id: str | None = None
    clarification: str | None = None  # non-null resumes a paused assessment


class AssessLoanResponse(BaseModel):
    thread_id: str
    final_assessment: str
    risk_level: str
    cited_articles: list[str]
    assessment_saved: bool


# ── Simple mutation request models ────────────────────────────────────────────

class UpdateStatusRequest(BaseModel):
    status: str


class UpdateCreditScoreRequest(BaseModel):
    credit_score: int


class PaymentCreateRequest(BaseModel):
    """Same as PaymentCreate but without customer_id (taken from path parameter)."""
    loan_id: UUID | None = None
    amount: Decimal
    due_date: date
    paid_date: date | None = None
    status: PaymentStatus
    notes: str | None = None