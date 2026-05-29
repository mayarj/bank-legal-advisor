import uuid
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.tools import BaseTool

from src.agents.legal_agent import build_legal_agent
from src.api.deps import get_tools
from src.api.schemas import AskRequest, AskResponse, ClarificationRequired

router = APIRouter(prefix="/ask", tags=["legal advisor"])


@router.post(
    "/",
    response_model=Union[AskResponse, ClarificationRequired],
    summary="Ask a legal question; the agent searches legislation and returns a cited answer",
)
async def ask_legal_question(
    body: AskRequest,
    tools: list[BaseTool] = Depends(get_tools),
):
    agent = build_legal_agent(tools)
    tid = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    if body.clarification and body.thread_id:
        from langgraph.types import Command
        result = await agent.ainvoke(Command(resume=body.clarification), config=config)
    else:
        result = await agent.ainvoke({"query": body.question, "messages": []}, config=config)

    if result.get("draft_answer"):
        return AskResponse(thread_id=tid, answer=result["draft_answer"])

    # Graph paused at an interrupt — surface the clarification question
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