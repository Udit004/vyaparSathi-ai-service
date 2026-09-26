"""
app/agent/subgraphs/document_generation/schemas.py
===================================================
Pydantic schemas for Document Generation specifications, validation, and tool inputs/outputs.
"""

from typing import Any, List, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Excel Schemas
# ---------------------------------------------------------------------------

class ExcelChartSpec(BaseModel):
    title: str = Field(..., description="Title of the chart")
    chart_type: Literal["bar", "line", "pie"] = Field("bar", description="Type of chart to render")
    sheet_name: str = Field(..., description="Target sheet containing data for the chart")
    categories_column: str = Field(..., description="Column header to use for category labels (X-axis)")
    values_column: str = Field(..., description="Column header to use for chart values (Y-axis)")


class ExcelSheetSpec(BaseModel):
    name: str = Field(..., description="Worksheet tab name (max 31 chars)")
    columns: List[str] = Field(..., description="List of column headers")
    rows: List[List[Any]] = Field(default_factory=list, description="Rows of data matching the order of columns")
    freeze_header: bool = Field(True, description="Whether to freeze the header row")
    auto_width: bool = Field(True, description="Whether to adjust column widths automatically")
    currency_columns: List[str] = Field(default_factory=list, description="Column names to format as currency")
    percentage_columns: List[str] = Field(default_factory=list, description="Column names to format as percentage")
    include_totals_row: bool = Field(False, description="Whether to append a total summary row at the bottom")


class ExcelDocumentSpec(BaseModel):
    format: Literal["xlsx"] = "xlsx"
    title: str = Field(..., description="Document main title")
    filename: Optional[str] = Field(None, description="Suggested human readable filename without directory")
    sheets: List[ExcelSheetSpec] = Field(..., description="List of worksheets")
    charts: List[ExcelChartSpec] = Field(default_factory=list, description="List of charts to include")


# ---------------------------------------------------------------------------
# Word Schemas
# ---------------------------------------------------------------------------

class WordTableSpec(BaseModel):
    headers: List[str] = Field(..., description="Table header titles")
    rows: List[List[Any]] = Field(default_factory=list, description="Table rows data")


class WordSectionSpec(BaseModel):
    heading: str = Field(..., description="Section title / heading text")
    heading_level: int = Field(1, description="Heading hierarchy level (1=H1, 2=H2, 3=H3)")
    content: Optional[str] = Field(None, description="Section paragraph text")
    bullet_points: List[str] = Field(default_factory=list, description="Bullet point list items")
    numbered_points: List[str] = Field(default_factory=list, description="Numbered list items")
    table: Optional[WordTableSpec] = Field(None, description="Optional table embedded in this section")


class WordDocumentSpec(BaseModel):
    format: Literal["docx"] = "docx"
    title: str = Field(..., description="Document main title")
    subtitle: Optional[str] = Field(None, description="Optional subtitle or metadata header")
    filename: Optional[str] = Field(None, description="Suggested human readable filename without directory")
    sections: List[WordSectionSpec] = Field(..., description="Structured sections of the document")


# ---------------------------------------------------------------------------
# Request, Repair & Validation Schemas
# ---------------------------------------------------------------------------

class DocumentGenerationRequest(BaseModel):
    document_type: Literal["excel", "word"] = Field(..., description="Target document type ('excel' or 'word')")
    title: str = Field(..., description="Title or subject of the document")
    purpose: Optional[str] = Field(None, description="Objective or context for document generation")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements or formatting rules")
    data: Optional[Any] = Field(None, description="Business data to be presented in the document")


class RepairInstructions(BaseModel):
    repair_required: bool = Field(False, description="Whether repair is necessary")
    errors: List[str] = Field(default_factory=list, description="Errors observed during previous attempt")
    repair_actions: List[str] = Field(default_factory=list, description="Specific adjustments to apply to specification")


class ValidationOutput(BaseModel):
    valid: bool = Field(..., description="True if document passed all validation checks")
    errors: List[str] = Field(default_factory=list, description="Validation failure details")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")


# ---------------------------------------------------------------------------
# Tool Input / Output Schemas
# ---------------------------------------------------------------------------

class DocumentGenerationInput(BaseModel):
    document_type: Literal["excel", "word"] = Field(..., description="Format type: 'excel' or 'word'")
    title: str = Field(..., description="Title of the report or document")
    purpose: Optional[str] = Field(None, description="Purpose or description of the document")
    requirements: Optional[List[str]] = Field(default_factory=list, description="Requirements or notes")
    data: Optional[Any] = Field(None, description="Structured business data (e.g. products, sales summary)")


class FileMetadataResult(BaseModel):
    file_id: str
    filename: str
    mime_type: str
    size: int
    storage_key: str
    document_type: str
    status: str
    download_url: Optional[str] = None


class DocumentToolOutput(BaseModel):
    success: bool
    file: Optional[FileMetadataResult] = None
    document_type: str
    attempts: int
    validation: ValidationOutput
    error: Optional[dict[str, Any]] = None
