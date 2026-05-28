import json
import uuid

from src.mcp.app import mcp
from src.db.session import AsyncSessionFactory
from src.db.schemas import (
    AssessmentCreate,
    CollateralType,
    LoanCreate,
    LoanStatus,
    LoanType,
    RiskLevel,
)
from src.db.crud import (
    create_loan,
    get_loan,
    list_loans as _list_loans,
    update_loan_status as _update_loan_status,
    delete_loan as _delete_loan,
    create_assessment,
    get_assessments_for_loan,
    get_latest_assessment,
)
from src.db.models import Loan, LoanAssessment


@mcp.tool()
async def create_loan_record(
    applicant_name: str,
    loan_type: str,
    amount: float,
    purpose: str,
    duration_months: int,
    interest_rate: float,
    collateral_type: str | None = None,
    applicant_credit_score: int | None = None,
    customer_id: str | None = None,
) -> str:
    """Create a new loan application record.
    loan_type: personal | mortgage | business | auto
    collateral_type: real_estate | vehicle | savings | none (omit if no collateral)
    customer_id: UUID of an existing CustomerProfile to link this loan to (optional)"""
    data = LoanCreate(
        applicant_name=applicant_name,
        loan_type=LoanType(loan_type),
        amount=amount,
        purpose=purpose,
        collateral_type=CollateralType(collateral_type) if collateral_type else None,
        duration_months=duration_months,
        interest_rate=interest_rate,
        applicant_credit_score=applicant_credit_score,
        customer_id=uuid.UUID(customer_id) if customer_id else None,
    )
    async with AsyncSessionFactory() as session:
        loan = await create_loan(session, data)
        await session.commit()
        return f"Loan created.\n{_format_loan(loan)}"


@mcp.tool()
async def get_loan_details(loan_id: str) -> str:
    """Retrieve the full details of a loan application and its latest assessment (if any)."""
    async with AsyncSessionFactory() as session:
        loan = await get_loan(session, uuid.UUID(loan_id))
        if loan is None:
            return f"Loan {loan_id} not found."
        latest = await get_latest_assessment(session, loan.id)
        result = _format_loan(loan)
        if latest:
            result += f"\n\nLatest Assessment:\n{_format_assessment(latest)}"
        return result


@mcp.tool()
async def list_loan_records(status: str | None = None) -> str:
    """List loan applications, optionally filtered by status.
    status: pending | under_review | approved | rejected (omit for all)"""
    async with AsyncSessionFactory() as session:
        loans = await _list_loans(
            session,
            status=LoanStatus(status) if status else None,
        )
        if not loans:
            return "No loans found."
        return "\n\n---\n\n".join(_format_loan(loan) for loan in loans)


@mcp.tool()
async def update_loan_status(loan_id: str, new_status: str) -> str:
    """Update the status of a loan application.
    new_status: pending | under_review | approved | rejected"""
    async with AsyncSessionFactory() as session:
        loan = await _update_loan_status(session, uuid.UUID(loan_id), LoanStatus(new_status))
        if loan is None:
            return f"Loan {loan_id} not found."
        await session.commit()
        return f"Status updated.\n{_format_loan(loan)}"


@mcp.tool()
async def save_assessment(
    loan_id: str,
    legal_question: str,
    answer: str,
    risk_level: str,
    cited_articles: str = "[]",
) -> str:
    """Save a legal assessment result for a loan application.
    risk_level: low | medium | high
    cited_articles: JSON array string, e.g. '["LAW-88-2003 | Article 2"]'"""
    try:
        articles = json.loads(cited_articles)
    except json.JSONDecodeError:
        articles = []

    data = AssessmentCreate(
        loan_id=uuid.UUID(loan_id),
        legal_question=legal_question,
        answer=answer,
        cited_articles=articles,
        risk_level=RiskLevel(risk_level),
    )
    async with AsyncSessionFactory() as session:
        assessment = await create_assessment(session, data)
        await session.commit()
        return f"Assessment saved.\n{_format_assessment(assessment)}"


@mcp.tool()
async def get_loan_assessments(loan_id: str) -> str:
    """Retrieve the full assessment history for a loan, newest first."""
    async with AsyncSessionFactory() as session:
        assessments = await get_assessments_for_loan(session, uuid.UUID(loan_id))
        if not assessments:
            return f"No assessments found for loan {loan_id}."
        return "\n\n---\n\n".join(_format_assessment(a) for a in assessments)


@mcp.tool()
async def delete_loan_record(loan_id: str) -> str:
    """Permanently delete a loan application and all its assessments."""
    async with AsyncSessionFactory() as session:
        deleted = await _delete_loan(session, uuid.UUID(loan_id))
        if not deleted:
            return f"Loan {loan_id} not found."
        await session.commit()
        return f"Loan {loan_id} deleted."


# ── Private formatters ────────────────────────────────────────────────────────

def _format_loan(loan: Loan) -> str:
    collateral = loan.collateral_type.value if loan.collateral_type else "none"
    credit = str(loan.applicant_credit_score) if loan.applicant_credit_score else "N/A"
    customer = str(loan.customer_id) if loan.customer_id else "none"
    return (
        f"ID: {loan.id}\n"
        f"Applicant: {loan.applicant_name}\n"
        f"Type: {loan.loan_type.value} | Amount: {loan.amount} | Duration: {loan.duration_months} months\n"
        f"Interest Rate: {loan.interest_rate}% | Collateral: {collateral} | Credit Score: {credit}\n"
        f"Purpose: {loan.purpose}\n"
        f"Customer Profile ID: {customer}\n"
        f"Status: {loan.status.value} | Created: {loan.created_at}"
    )


def _format_assessment(assessment: LoanAssessment) -> str:
    try:
        articles = json.loads(assessment.cited_articles)
    except (json.JSONDecodeError, TypeError):
        articles = []
    articles_text = ", ".join(articles) if articles else "none"
    return (
        f"Risk Level: {assessment.risk_level.value}\n"
        f"Cited Articles: {articles_text}\n"
        f"Question: {assessment.legal_question}\n"
        f"Answer: {assessment.answer}"
    )