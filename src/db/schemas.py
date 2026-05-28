from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    AMENDS = "amends"
    REPEALS = "repeals"
    SUPERSEDES = "supersedes"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    CONFLICTS_WITH = "conflicts_with"


class LegislationStatus(str, Enum):
    ACTIVE = "active"
    REPEALED = "repealed"
    AMENDED = "amended"
    PENDING = "pending"
    DRAFT = "draft"


class Relationship(BaseModel):
    type: RelationshipType
    father_legislation: str
    father_article: Optional[str] = None
    affected_legislation: str
    affected_article: Optional[str] = None
    illustration: str = Field(max_length=150)


class Legislation(BaseModel):
    code: str
    date: date
    issuer: str
    subject: str = Field(max_length=200)
    status: LegislationStatus
    articles: dict[str, str] = Field(default_factory=dict)
    relationships: list[Relationship] = Field(default_factory=list)


# ── Customer schemas ──────────────────────────────────────────────────────────

class EmploymentStatus(str, Enum):
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"


class PaymentStatus(str, Enum):
    ON_TIME = "on_time"
    LATE = "late"
    MISSED = "missed"
    PARTIAL = "partial"


class CustomerProfileCreate(BaseModel):
    national_id: str
    full_name: str
    date_of_birth: Optional[date] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    employment_status: EmploymentStatus
    monthly_income: Optional[Decimal] = None
    existing_debt_amount: Decimal = Decimal("0.00")
    credit_score: Optional[int] = None  # 300–850


class PaymentCreate(BaseModel):
    customer_id: UUID
    loan_id: Optional[UUID] = None
    amount: Decimal
    due_date: date
    paid_date: Optional[date] = None
    status: PaymentStatus
    notes: Optional[str] = None


# ── Loan schemas ──────────────────────────────────────────────────────────────

class LoanType(str, Enum):
    PERSONAL = "personal"
    MORTGAGE = "mortgage"
    BUSINESS = "business"
    AUTO = "auto"


class LoanStatus(str, Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CollateralType(str, Enum):
    REAL_ESTATE = "real_estate"
    VEHICLE = "vehicle"
    SAVINGS = "savings"
    NONE = "none"


class LoanCreate(BaseModel):
    applicant_name: str
    loan_type: LoanType
    amount: Decimal
    purpose: str
    collateral_type: Optional[CollateralType] = None
    duration_months: int
    interest_rate: Decimal
    applicant_credit_score: Optional[int] = None
    customer_id: Optional[UUID] = None


class AssessmentCreate(BaseModel):
    loan_id: UUID
    legal_question: str
    answer: str
    cited_articles: list[str] = Field(default_factory=list)
    risk_level: RiskLevel
