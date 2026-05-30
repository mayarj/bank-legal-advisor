import sys
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

_SERVER_CONFIG = {
    "legal-server": {
        "command": sys.executable,
        "args": ["-m", "src.mcp.server"],
        "transport": "stdio",
    }
}


@asynccontextmanager
async def mcp_client():
    """Start the MCP server subprocess, yield LangChain-compatible tools,
    and keep the connection alive until the context exits.

    langchain-mcp-adapters >= 0.1.0 removed the async-context-manager interface
    from MultiServerMCPClient. The client object must stay in scope for the full
    lifetime of tool use — moving it to a local variable inside the generator
    achieves this because generator locals live until the generator is exhausted.

    Usage (FastAPI lifespan):
        async with mcp_client() as tools:
            app.state.tools = tools
            yield
    """
    client = MultiServerMCPClient(_SERVER_CONFIG)
    tools = await client.get_tools()
    yield tools
    # generator returns here → client goes out of scope → subprocess terminates


async def get_tools_once() -> list[BaseTool]:
    """Return tools without holding the connection open.
    Only safe for inspection or pre-binding — not for actual agent execution."""
    client = MultiServerMCPClient(_SERVER_CONFIG)
    return await client.get_tools()