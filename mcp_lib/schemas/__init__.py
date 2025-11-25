"""MCP schemas for input/output."""

from pydantic import BaseModel


class MCPInput(BaseModel):
    """Input schema for MCP agent requests."""

    question: str


class MCPOutput(BaseModel):
    """Output schema for MCP agent responses."""

    answer: str

