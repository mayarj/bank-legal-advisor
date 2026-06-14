import uuid
from typing import Union

from fastapi import APIRouter, Depends
from langchain_core.tools import BaseTool

from src.agents.legal_agent import build_legal_agent
from src.api.deps import get_checkpointer, get_tools
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
    checkpointer=Depends(get_checkpointer),
):
    agent = build_legal_agent(tools, checkpointer=checkpointer)
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
    message, options = _first_interrupt(snapshot)
    return ClarificationRequired(thread_id=tid, question=message, options=options)


def _first_interrupt(snapshot) -> tuple[str, list]:
    for task in snapshot.tasks:
        for intr in getattr(task, "interrupts", []):
            value = intr.value if isinstance(intr.value, dict) else {}
            return (
                str(value.get("message", "Please provide more information.")),
                value.get("options", []) or [],
            )
    return "Please provide more information.", []