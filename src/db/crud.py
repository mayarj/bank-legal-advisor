import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Article as ArticleModel,
    CustomerProfile,
    Legislation as LegislationModel,
    Loan,
    LoanAssessment,
    Payment,
    Relationship as RelationshipModel,
)
from src.db.schemas import (
    AssessmentCreate,
    CustomerProfileCreate,
    Legislation as LegislationSchema,
    LoanCreate,
    LoanStatus,
    PaymentCreate,
)
from src.rag.status_policy import article_baseline


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


# ── Legislation & Article (status reconciliation) ──────────────────────────────

def _article_id(legislation_code: str, article_number: str) -> str:
    return f"{legislation_code}_article_{article_number}"


async def upsert_legislation(
    session: AsyncSession, legislation: LegislationSchema
) -> LegislationModel:
    """Create or update the legislation metadata row (idempotent re-ingestion)."""
    row = await session.get(LegislationModel, legislation.code)
    if row is None:
        row = LegislationModel(code=legislation.code)
        session.add(row)
    row.date = legislation.date
    row.issuer = legislation.issuer
    row.subject = legislation.subject
    row.status = legislation.status
    await session.flush()
    return row


async def upsert_articles(
    session: AsyncSession, legislation: LegislationSchema
) -> list[ArticleModel]:
    """Create Article rows for a legislation's articles, seeded from the legislation's
    baseline status. Existing rows are left untouched so a previously reconciled status
    is not clobbered — reconciliation recomputes them afterwards."""
    base_status, base_in_force = article_baseline(legislation.status)
    rows: list[ArticleModel] = []
    for number in legislation.articles:
        aid = _article_id(legislation.code, number)
        row = await session.get(ArticleModel, aid)
        if row is None:
            row = ArticleModel(
                id=aid,
                legislation_code=legislation.code,
                article_number=number,
                status=base_status,
                is_in_force=base_in_force,
            )
            session.add(row)
        rows.append(row)
    await session.flush()
    return rows


async def get_legislation_row(
    session: AsyncSession, legislation_code: str
) -> Optional[LegislationModel]:
    return await session.get(LegislationModel, legislation_code)


async def get_article_row(
    session: AsyncSession, legislation_code: str, article_number: str
) -> Optional[ArticleModel]:
    return await session.get(ArticleModel, _article_id(legislation_code, article_number))


async def get_articles_for_legislation(
    session: AsyncSession, legislation_code: str
) -> list[ArticleModel]:
    result = await session.execute(
        select(ArticleModel).where(ArticleModel.legislation_code == legislation_code)
    )
    return list(result.scalars().all())


async def get_incoming_relationships(
    session: AsyncSession, legislation_code: str, article_number: str
) -> list[RelationshipModel]:
    """Every relationship that targets this article — including whole-legislation
    relationships (affected_article IS NULL), which apply to all of its articles."""
    result = await session.execute(
        select(RelationshipModel).where(
            RelationshipModel.affected_legislation == legislation_code,
            or_(
                RelationshipModel.affected_article == article_number,
                RelationshipModel.affected_article.is_(None),
            ),
        )
    )
    return list(result.scalars().all())