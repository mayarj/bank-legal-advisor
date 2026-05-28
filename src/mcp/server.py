from src.mcp.app import mcp

from src.mcp.tools import legal_lookup
from src.mcp.tools import loans
from src.mcp.tools import customers

if __name__ == "__main__":
    mcp.run(transport="stdio")