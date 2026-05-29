from fastapi import Request
from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db  # re-exported for route convenience


async def get_tools(request: Request) -> list[BaseTool]:
    """Inject the MCP tool list that was bound to app.state at startup."""
    return request.app.state.tools