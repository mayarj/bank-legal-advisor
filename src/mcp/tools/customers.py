import uuid
from collections import Counter

from src.mcp.app import mcp
from src.db.session import AsyncSessionFactory
from src.db.crud import (
    get_customer_profile as _get_customer_profile,
    get_customer_by_national_id as _get_customer_by_national_id,
    get_payments_for_customer,
    get_loans_for_customer,
)
from src.db.models import CustomerProfile, Loan, Payment


@mcp.tool()
async def get_customer_profile_by_id(customer_id: str) -> str:
    """Retrieve a customer's full credit profile by their UUID.
    Returns identity, employment, income, existing debt, and credit score.
    Use before assessing a loan to understand the applicant's financial standing."""
    async with AsyncSessionFactory() as session:
        profile = await _get_customer_profile(session, uuid.UUID(customer_id))
        if profile is None:
            return f"Customer {customer_id} not found."
        return _format_profile(profile)


@mcp.tool()
async def get_customer_profile_by_national_id(national_id: str) -> str:
    """Retrieve a customer's full credit profile by their national ID number.
    Use when you have the national ID but not the internal UUID."""
    async with AsyncSessionFactory() as session:
        profile = await _get_customer_by_national_id(session, national_id)
        if profile is None:
            return f"No customer found with national ID '{national_id}'."
        return _format_profile(profile)


@mcp.tool()
async def get_customer_payment_history(customer_id: str) -> str:
    """Retrieve the full payment history for a customer, newest due-date first.
    Shows each payment's amount, due date, paid date, and status (on_time | late | missed | partial).
    Use to evaluate repayment behaviour during loan risk assessment."""
    async with AsyncSessionFactory() as session:
        payments = await get_payments_for_customer(session, uuid.UUID(customer_id))
        if not payments:
            return f"No payment records found for customer {customer_id}."

        counts = Counter(p.status.value for p in payments)
        summary = (
            f"Payment summary — "
            f"On-time: {counts.get('on_time', 0)} | "
            f"Late: {counts.get('late', 0)} | "
            f"Missed: {counts.get('missed', 0)} | "
            f"Partial: {counts.get('partial', 0)} | "
            f"Total: {len(payments)}"
        )
        records = "\n".join(_format_payment(p) for p in payments)
        return f"{summary}\n\n{records}"


@mcp.tool()
async def get_customer_loans(customer_id: str) -> str:
    """Retrieve all loans linked to a customer, newest first.
    Shows loan type, amount, status, interest rate, and collateral.
    Use to check existing debt obligations during loan risk assessment."""
    async with AsyncSessionFactory() as session:
        loans = await get_loans_for_customer(session, uuid.UUID(customer_id))
        if not loans:
            return f"No loans found for customer {customer_id}."
        return "\n\n---\n\n".join(_format_loan(loan) for loan in loans)


# ── Private formatters ────────────────────────────────────────────────────────

def _format_profile(p: CustomerProfile) -> str:
    income = f"{p.monthly_income}" if p.monthly_income is not None else "N/A"
    score = str(p.credit_score) if p.credit_score is not None else "N/A"
    score_date = p.credit_score_updated_at.date() if p.credit_score_updated_at else "N/A"
    dob = str(p.date_of_birth) if p.date_of_birth else "N/A"
    dti = None
    if p.monthly_income and p.existing_debt_amount:
        dti = f"{(p.existing_debt_amount / p.monthly_income * 100):.1f}%"

    lines = [
        f"ID: {p.id}",
        f"Name: {p.full_name} | National ID: {p.national_id}",
        f"Date of Birth: {dob} | Phone: {p.phone or 'N/A'} | Email: {p.email or 'N/A'}",
        f"Employment: {p.employment_status.value} | Monthly Income: {income}",
        f"Existing Debt: {p.existing_debt_amount}"
        + (f" | Debt-to-Income: {dti}" if dti else ""),
        f"Credit Score: {score} (as of {score_date})",
    ]
    return "\n".join(lines)


def _format_payment(p: Payment) -> str:
    paid = str(p.paid_date) if p.paid_date else "not paid"
    loan_ref = f"Loan {p.loan_id}" if p.loan_id else "no loan ref"
    return (
        f"  [{p.status.value.upper()}] Amount: {p.amount} | "
        f"Due: {p.due_date} | Paid: {paid} | {loan_ref}"
        + (f" | Notes: {p.notes}" if p.notes else "")
    )


def _format_loan(loan: Loan) -> str:
    collateral = loan.collateral_type.value if loan.collateral_type else "none"
    return (
        f"Loan ID: {loan.id}\n"
        f"Type: {loan.loan_type.value} | Amount: {loan.amount} | Status: {loan.status.value}\n"
        f"Duration: {loan.duration_months} months | Interest Rate: {loan.interest_rate}% | "
        f"Collateral: {collateral}\n"
        f"Purpose: {loan.purpose} | Created: {loan.created_at}"
    )