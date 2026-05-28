import sys
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

# Starts the MCP server as a subprocess using the same Python interpreter.
# stdio transport: client writes to the server's stdin, reads from its stdout.
_SERVER_CONFIG = {
    "legal-server": {
        "command": sys.executable,
        "args": ["-m", "src.mcp.server"],
        "transport": "stdio",
    }
}


@asynccontextmanager
async def mcp_client():
    """Async context manager that starts the MCP server, yields LangChain-compatible
    tools, and shuts the server down on exit.

    The connection must stay open while tools are in use — tools call back to the
    server subprocess when invoked, so closing the context before the agent finishes
    will break in-flight tool calls.

    Usage in an agent:
        async with mcp_client() as tools:
            agent = create_react_agent(llm, tools)
            response = await agent.ainvoke({...})

    Usage in FastAPI lifespan (keeps connection alive for all requests):
        async with mcp_client() as tools:
            app.state.tools = tools
            yield
    """
    async with MultiServerMCPClient(_SERVER_CONFIG) as client:
        yield client.get_tools()


async def get_tools_once() -> list[BaseTool]:
    """Returns tools without holding the connection open.
    Only safe when the tools themselves do not call back to the server after this
    function returns — i.e. for inspection, testing, or pre-binding to an LLM.
    For actual agent execution use mcp_client() instead."""
    async with MultiServerMCPClient(_SERVER_CONFIG) as client:
        return client.get_tools()