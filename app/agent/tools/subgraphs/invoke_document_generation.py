"""
app/agent/tools/subgraphs/invoke_document_generation.py
========================================================
Tool stub that signals the subgraph_router to execute the Document Generation Subgraph.
"""

import ast
import json
from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool


class InvokeDocumentGenerationInput(BaseModel):
    document_type: Literal["excel", "word"] = Field(..., description="Target document type: 'excel' or 'word'.")
    title: str = Field(..., description="Title of the report or document.")
    purpose: Optional[str] = Field(None, description="Purpose or description of what the document should present.")
    requirements: Optional[Union[List[str], str]] = Field(default_factory=list, description="Specific formatting or layout requirements.")
    data: Optional[Any] = Field(None, description="Business data (e.g. products list, sales summary) to include in the document.")

    @field_validator("requirements", mode="before")
    @classmethod
    def parse_requirements(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    parsed = json.loads(v_str)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    try:
                        parsed = ast.literal_eval(v_str)
                        if isinstance(parsed, list):
                            return [str(x) for x in parsed]
                    except Exception:
                        pass
            return [v_str]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]

    @field_validator("data", mode="before")
    @classmethod
    def parse_data(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_str = v.strip()
            if (v_str.startswith("{") and v_str.endswith("}")) or (v_str.startswith("[") and v_str.endswith("]")):
                try:
                    return json.loads(v_str)
                except Exception:
                    try:
                        return ast.literal_eval(v_str)
                    except Exception:
                        pass
        return v


@tool("invoke_document_generation", args_schema=InvokeDocumentGenerationInput)
async def invoke_document_generation(
    document_type: str,
    title: str,
    purpose: Optional[str] = None,
    requirements: Optional[Union[List[str], str]] = None,
    data: Optional[Any] = None,
) -> dict:

    """
    [SUBGRAPH TOOL] Generate and validate a business document (Excel .xlsx or Word .docx) and upload to Cloudflare R2 storage.

    Use when the user requests a downloadable file or report:
      - "Generate an Excel report of my low stock products"
      - "Create a Word document for monthly sales overview"
      - "Export inventory data into an Excel spreadsheet"
      - "Create a business report for my store"

    OUTPUT:
      success      : bool   — True if document generated, validated, and uploaded
      file         : dict   — {file_id, filename, mime_type, size, storage_key, download_url}
      document_type: str    — "excel" or "word"
      attempts     : int    — Number of generation attempts
      validation   : dict   — Validation status
    """
    return {
        "__subgraph__": "document_generation",
        "request": {
            "document_type": document_type,
            "title": title,
            "purpose": purpose,
            "requirements": requirements or [],
            "data": data,
        },
        "document_type": document_type,
        "source_data": data,
    }
