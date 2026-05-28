from src.mcp.app import mcp  # noqa: F401

# Tools are imported AFTER mcp is defined so their decorators can register against it
from src.mcp.tools import legal_lookup  # noqa: E402, F401

if __name__ == "__main__":
    mcp.run(transport="stdio")