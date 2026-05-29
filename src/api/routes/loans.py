import uuid
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.loan_agent import build_loan_agent
from src.api.deps import get_db, get_tools
from src.api.schemas import (
    AssessLoanRequest,
    AssessLoanResponse,
    ClarificationRequired,
    LoanResponse,
    UpdateStatusRequest,
)
from src.db.crud import (
    create_loan,
    delete_loan,
    get_loan,
    list_loans,
    update_loan_status,
)
from src.db.schemas import LoanCreate, LoanStatus

router = APIRouter(prefix="/loans", tags=["loans"])


@router.post("/", response_model=LoanResponse, status_code=201)
async def create_loan_endpoint(
    body: LoanCreate,
    session: AsyncSession = Depends(get_db),
):
    loan = await create_loan(session, body)
    return loan


@router.get("/", response_model=list[LoanResponse])
async def list_loans_endpoint(
    status: str | None = None,
    session: AsyncSession = Depends(get_db),
):
    try:
        status_enum = LoanStatus(status) if status else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status '{status}'.")
    return await list_loans(session, status=status_enum)


@router.get("/{loan_id}", response_model=LoanResponse)
async def get_loan_endpoint(
    loan_id: str,
    session: AsyncSession = Depends(get_db),
):
    loan = await get_loan(session, uuid.UUID(loan_id))
    if loan is None:
        raise HTTPException(status_code=404, detail=f"Loan {loan_id} not found.")
    return loan


@router.put("/{loan_id}/status", response_model=LoanResponse)
async def update_status_endpoint(
    loan_id: str,
    body: UpdateStatusRequest,
    session: AsyncSession = Depends(get_db),
):
    try:
        status_enum = LoanStatus(body.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status '{body.status}'.")
    loan = await update_loan_status(session, uuid.UUID(loan_id), status_enum)
    if loan is None:
        raise HTTPException(status_code=404, detail=f"Loan {loan_id} not found.")
    return loan


@router.delete("/{loan_id}", status_code=200)
async def delete_loan_endpoint(
    loan_id: str,
    session: AsyncSession = Depends(get_db),
):
    deleted = await delete_loan(session, uuid.UUID(loan_id))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Loan {loan_id} not found.")
    return {"deleted": True, "loan_id": loan_id}


@router.post(
    "/{loan_id}/assess",
    response_model=Union[AssessLoanResponse, ClarificationRequired],
    summary="Run the loan assessment agent; returns the assessment or a clarification request",
)
async def assess_loan_endpoint(
    loan_id: str,
    body: AssessLoanRequest = AssessLoanRequest(),
    tools: list[BaseTool] = Depends(get_tools),
    session: AsyncSession = Depends(get_db),
):
    loan = await get_loan(session, uuid.UUID(loan_id))
    if loan is None:
        raise HTTPException(status_code=404, detail=f"Loan {loan_id} not found.")

    agent = build_loan_agent(tools)
    tid = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    if body.clarification and body.thread_id:
        from langgraph.types import Command
        result = await agent.ainvoke(Command(resume=body.clarification), config=config)
    else:
        result = await agent.ainvoke({"loan_id": loan_id, "messages": []}, config=config)

    if result.get("assessment_saved"):
        return AssessLoanResponse(
            thread_id=tid,
            final_assessment=result["final_assessment"],
            risk_level=result["risk_level"],
            cited_articles=result.get("cited_articles") or [],
            assessment_saved=True,
        )

    # Agent paused waiting for clarification
    snapshot = await agent.aget_state(config)
    return ClarificationRequired(
        thread_id=tid,
        question=_first_interrupt_message(snapshot),
    )


def _first_interrupt_message(snapshot) -> str:
    for task in snapshot.tasks:
        for intr in getattr(task, "interrupts", []):
            return str(intr.value.get("message", "Please provide more information."))
    return "Please provide more information."