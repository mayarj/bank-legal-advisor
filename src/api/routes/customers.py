import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.schemas import (
    CustomerResponse,
    LoanResponse,
    PaymentCreateRequest,
    PaymentResponse,
    UpdateCreditScoreRequest,
)
from src.db.crud import (
    create_customer_profile,
    create_payment,
    get_customer_by_national_id,
    get_customer_profile,
    get_loans_for_customer,
    get_payments_for_customer,
    list_customer_profiles,
    update_customer_credit_score,
)
from src.db.schemas import CustomerProfileCreate, PaymentCreate

router = APIRouter(prefix="/customers", tags=["customers"])


@router.post("/", response_model=CustomerResponse, status_code=201)
async def create_customer_endpoint(
    body: CustomerProfileCreate,
    session: AsyncSession = Depends(get_db),
):
    profile = await create_customer_profile(session, body)
    return profile


@router.get("/", response_model=list[CustomerResponse])
async def list_customers_endpoint(session: AsyncSession = Depends(get_db)):
    return await list_customer_profiles(session)


# NOTE: this route must be defined before /{customer_id} to avoid path shadowing
@router.get("/by-national-id/{national_id}", response_model=CustomerResponse)
async def get_customer_by_national_id_endpoint(
    national_id: str,
    session: AsyncSession = Depends(get_db),
):
    profile = await get_customer_by_national_id(session, national_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No customer with national ID '{national_id}'.")
    return profile


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer_endpoint(
    customer_id: str,
    session: AsyncSession = Depends(get_db),
):
    profile = await get_customer_profile(session, uuid.UUID(customer_id))
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")
    return profile


@router.put("/{customer_id}/credit-score", response_model=CustomerResponse)
async def update_credit_score_endpoint(
    customer_id: str,
    body: UpdateCreditScoreRequest,
    session: AsyncSession = Depends(get_db),
):
    if not (300 <= body.credit_score <= 850):
        raise HTTPException(status_code=400, detail="Credit score must be between 300 and 850.")
    profile = await update_customer_credit_score(session, uuid.UUID(customer_id), body.credit_score)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")
    return profile


@router.get("/{customer_id}/loans", response_model=list[LoanResponse])
async def get_customer_loans_endpoint(
    customer_id: str,
    session: AsyncSession = Depends(get_db),
):
    await _require_customer(session, customer_id)
    return await get_loans_for_customer(session, uuid.UUID(customer_id))


@router.get("/{customer_id}/payments", response_model=list[PaymentResponse])
async def get_customer_payments_endpoint(
    customer_id: str,
    session: AsyncSession = Depends(get_db),
):
    await _require_customer(session, customer_id)
    return await get_payments_for_customer(session, uuid.UUID(customer_id))


@router.post("/{customer_id}/payments", response_model=PaymentResponse, status_code=201)
async def add_payment_endpoint(
    customer_id: str,
    body: PaymentCreateRequest,
    session: AsyncSession = Depends(get_db),
):
    await _require_customer(session, customer_id)
    data = PaymentCreate(customer_id=uuid.UUID(customer_id), **body.model_dump())
    payment = await create_payment(session, data)
    return payment


# ── Helper ─────────────────────────────────────────────────────────────────────

async def _require_customer(session: AsyncSession, customer_id: str):
    profile = await get_customer_profile(session, uuid.UUID(customer_id))
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")