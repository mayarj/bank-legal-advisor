import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Date, DateTime, Enum as SAEnum, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.db.base import Base
from src.db.schemas import (
    CollateralType,
    EmploymentStatus,
    LoanStatus,
    LoanType,
    PaymentStatus,
    RelationshipType,
    RiskLevel,
)


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    type: Mapped[RelationshipType] = mapped_column(SAEnum(RelationshipType))
    father_legislation: Mapped[str] = mapped_column(String)
    father_article: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    affected_legislation: Mapped[str] = mapped_column(String)
    affected_article: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    illustration: Mapped[str] = mapped_column(String(150))

    __table_args__ = (
        Index("ix_father_legislation", "father_legislation"),
        Index("ix_affected_legislation", "affected_legislation"),
    )


class CustomerProfile(Base):
    __tablename__ = "customer_profiles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    national_id: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[str] = mapped_column(String)
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    employment_status: Mapped[EmploymentStatus] = mapped_column(SAEnum(EmploymentStatus))
    monthly_income: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2), nullable=True)
    existing_debt_amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=Decimal("0.00"))
    credit_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    credit_score_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_customer_national_id", "national_id", unique=True),
    )


class Loan(Base):
    __tablename__ = "loans"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    applicant_name: Mapped[str] = mapped_column(String)
    loan_type: Mapped[LoanType] = mapped_column(SAEnum(LoanType))
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    purpose: Mapped[str] = mapped_column(String)
    status: Mapped[LoanStatus] = mapped_column(SAEnum(LoanStatus), default=LoanStatus.PENDING)
    collateral_type: Mapped[Optional[CollateralType]] = mapped_column(SAEnum(CollateralType), nullable=True)
    duration_months: Mapped[int] = mapped_column(Integer)
    interest_rate: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    applicant_credit_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    customer_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("customer_profiles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_loan_status", "status"),
    )


class LoanAssessment(Base):
    __tablename__ = "loan_assessments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    loan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("loans.id", ondelete="CASCADE"), index=True
    )
    legal_question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    cited_articles: Mapped[str] = mapped_column(Text)  # stored as JSON string
    risk_level: Mapped[RiskLevel] = mapped_column(SAEnum(RiskLevel))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    customer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("customer_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    loan_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("loans.id", ondelete="SET NULL"), nullable=True, index=True
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    due_date: Mapped[date] = mapped_column(Date)
    paid_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[PaymentStatus] = mapped_column(SAEnum(PaymentStatus))
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_payment_status", "status"),
    )