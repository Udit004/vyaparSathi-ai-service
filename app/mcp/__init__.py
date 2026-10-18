"""
Vyapar Sathi MCP (Model Context Protocol) Server package.
Provides read-only tools for Claude, ChatGPT, Cursor, and other MCP clients.
"""
from app.mcp.server import mcp

__all__ = ["mcp"]
