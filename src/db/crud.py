import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CustomerProfile, Loan, LoanAssessment, Payment
from src.db.schemas import AssessmentCreate, CustomerProfileCreate, LoanCreate, LoanStatus, PaymentCreate


# ── Loan ──────────────────────────────────────────────────────────────────────

async def create_loan(session: AsyncSession, data: LoanCreate) -> Loan:
    loan = Loan(
        id=uuid.uuid4(),
        applicant_name=data.applicant_name,
        loan_type=data.loan_type,
        amount=data.amount,
        purpose=data.purpose,
        status=LoanStatus.PENDING,
        collateral_type=data.collateral_type,
        duration_months=data.duration_months,
        interest_rate=data.interest_rate,
        applicant_credit_score=data.applicant_credit_score,
        customer_id=data.customer_id,
    )
    session.add(loan)
    await session.flush()
    return loan


async def get_loan(session: AsyncSession, loan_id: uuid.UUID) -> Optional[Loan]:
    result = await session.execute(select(Loan).where(Loan.id == loan_id))
    return result.scalars().first()


async def list_loans(
    session: AsyncSession,
    status: Optional[LoanStatus] = None,
    limit: int = 50,
) -> list[Loan]:
    stmt = select(Loan)
    if status is not None:
        stmt = stmt.where(Loan.status == status)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_loans_for_customer(
    session: AsyncSession,
    customer_id: uuid.UUID,
) -> list[Loan]:
    result = await session.execute(
        select(Loan)
        .where(Loan.customer_id == customer_id)
        .order_by(Loan.created_at.desc())
    )
    return list(result.scalars().all())


async def update_loan_status(
    session: AsyncSession,
    loan_id: uuid.UUID,
    status: LoanStatus,
) -> Optional[Loan]:
    loan = await get_loan(session, loan_id)
    if loan is None:
        return None
    loan.status = status
    await session.flush()
    return loan


async def delete_loan(session: AsyncSession, loan_id: uuid.UUID) -> bool:
    loan = await get_loan(session, loan_id)
    if loan is None:
        return False
    await session.delete(loan)
    await session.flush()
    return True


# ── LoanAssessment ────────────────────────────────────────────────────────────

async def create_assessment(session: AsyncSession, data: AssessmentCreate) -> LoanAssessment:
    assessment = LoanAssessment(
        id=uuid.uuid4(),
        loan_id=data.loan_id,
        legal_question=data.legal_question,
        answer=data.answer,
        cited_articles=json.dumps(data.cited_articles),
        risk_level=data.risk_level,
    )
    session.add(assessment)
    await session.flush()
    return assessment


async def get_assessments_for_loan(
    session: AsyncSession,
    loan_id: uuid.UUID,
) -> list[LoanAssessment]:
    result = await session.execute(
        select(LoanAssessment)
        .where(LoanAssessment.loan_id == loan_id)
        .order_by(LoanAssessment.created_at.desc())
    )
    return list(result.scalars().all())


async def get_latest_assessment(
    session: AsyncSession,
    loan_id: uuid.UUID,
) -> Optional[LoanAssessment]:
    result = await session.execute(
        select(LoanAssessment)
        .where(LoanAssessment.loan_id == loan_id)
        .order_by(LoanAssessment.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


# ── CustomerProfile ───────────────────────────────────────────────────────────

async def create_customer_profile(
    session: AsyncSession, data: CustomerProfileCreate
) -> CustomerProfile:
    profile = CustomerProfile(
        id=uuid.uuid4(),
        national_id=data.national_id,
        full_name=data.full_name,
        date_of_birth=data.date_of_birth,
        phone=data.phone,
        email=data.email,
        employment_status=data.employment_status,
        monthly_income=data.monthly_income,
        existing_debt_amount=data.existing_debt_amount,
        credit_score=data.credit_score,
        credit_score_updated_at=datetime.now(timezone.utc) if data.credit_score else None,
    )
    session.add(profile)
    await session.flush()
    return profile


async def get_customer_profile(
    session: AsyncSession, customer_id: uuid.UUID
) -> Optional[CustomerProfile]:
    result = await session.execute(
        select(CustomerProfile).where(CustomerProfile.id == customer_id)
    )
    return result.scalars().first()


async def get_customer_by_national_id(
    session: AsyncSession, national_id: str
) -> Optional[CustomerProfile]:
    result = await session.execute(
        select(CustomerProfile).where(CustomerProfile.national_id == national_id)
    )
    return result.scalars().first()


async def list_customer_profiles(
    session: AsyncSession, limit: int = 50
) -> list[CustomerProfile]:
    result = await session.execute(select(CustomerProfile).limit(limit))
    return list(result.scalars().all())


async def update_customer_credit_score(
    session: AsyncSession,
    customer_id: uuid.UUID,
    credit_score: int,
) -> Optional[CustomerProfile]:
    profile = await get_customer_profile(session, customer_id)
    if profile is None:
        return None
    profile.credit_score = credit_score
    profile.credit_score_updated_at = datetime.now(timezone.utc)
    await session.flush()
    return profile


# ── Payment ───────────────────────────────────────────────────────────────────

async def create_payment(session: AsyncSession, data: PaymentCreate) -> Payment:
    payment = Payment(
        id=uuid.uuid4(),
        customer_id=data.customer_id,
        loan_id=data.loan_id,
        amount=data.amount,
        due_date=data.due_date,
        paid_date=data.paid_date,
        status=data.status,
        notes=data.notes,
    )
    session.add(payment)
    await session.flush()
    return payment


async def get_payments_for_customer(
    session: AsyncSession,
    customer_id: uuid.UUID,
    limit: int = 100,
) -> list[Payment]:
    result = await session.execute(
        select(Payment)
        .where(Payment.customer_id == customer_id)
        .order_by(Payment.due_date.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_payments_for_loan(
    session: AsyncSession,
    loan_id: uuid.UUID,
) -> list[Payment]:
    result = await session.execute(
        select(Payment)
        .where(Payment.loan_id == loan_id)
        .order_by(Payment.due_date.desc())
    )
    return list(result.scalars().all())