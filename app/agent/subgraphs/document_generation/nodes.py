"""
app/agent/subgraphs/document_generation/nodes.py
==================================================
Node implementations for the Document Generation Subgraph.
"""

import os
import tempfile
import uuid
from typing import Any
import structlog

from app.lib.llm import get_small_llm
from app.services.r2_storage_service import get_r2_storage_service
from app.agent.subgraphs.document_generation.state import DocumentGenerationState
from app.agent.subgraphs.document_generation.schemas import (
    ExcelDocumentSpec,
    WordDocumentSpec,
)
from app.agent.subgraphs.document_generation.generators.excel_generator import generate_excel_document
from app.agent.subgraphs.document_generation.generators.word_generator import generate_word_document
from app.agent.subgraphs.document_generation.validators.excel_validator import validate_excel_document
from app.agent.subgraphs.document_generation.validators.word_validator import validate_word_document

LOGGER = structlog.get_logger("vyaparsathi.ai.doc_gen.nodes")

TEMP_DOC_DIR = os.path.join(tempfile.gettempdir(), "vyapar_sathi_documents")


# ---------------------------------------------------------------------------
# Node 1: Prepare Document
# ---------------------------------------------------------------------------

async def prepare_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Initialize state defaults and normalize request inputs.
    """
    request = state.get("request", {})
    doc_type = state.get("document_type") or request.get("document_type", "excel")
    doc_type = doc_type.lower()
    if doc_type not in ("excel", "word"):
        doc_type = "excel"

    doc_format = "xlsx" if doc_type == "excel" else "docx"
    mime_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if doc_type == "excel"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    LOGGER.info(
        "doc_gen.prepare_document",
        doc_type=doc_type,
        doc_format=doc_format,
        store_id=state.get("store_id"),
    )

    return {
        "document_type": doc_type,
        "document_format": doc_format,
        "mime_type": mime_type,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
        "status": "in_progress",
        "validation_errors": [],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Node 2: Build Document Spec
# ---------------------------------------------------------------------------

def _fallback_excel_spec(title: str, data: Any) -> dict[str, Any]:
    """Construct a safe, valid default Excel specification directly from source data."""
    sheets = []

    if isinstance(data, dict):
        has_complex_nested = False
        for k, v in data.items():
            sheet_name = str(k).replace("_", " ").title()[:31]
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                has_complex_nested = True
                cols = [str(col).replace("_", " ").title() for col in v[0].keys()]
                rows = [[item.get(col) for col in v[0].keys()] for item in v]
                currency_cols = [c for c in cols if any(kw in c.lower() for kw in ["price", "revenue", "cost", "sales", "amount", "total"])]
                sheets.append({
                    "name": sheet_name,
                    "columns": cols,
                    "rows": rows,
                    "freeze_header": True,
                    "auto_width": True,
                    "currency_columns": currency_cols,
                    "include_totals_row": len(currency_cols) > 0,
                })
            elif isinstance(v, dict):
                has_complex_nested = True
                cols = ["Metric / Attribute", "Value"]
                rows = [[str(mk).replace("_", " ").title(), mv] for mk, mv in v.items()]
                currency_cols = [c for c in cols if any(kw in c.lower() for kw in ["price", "revenue", "cost", "sales", "amount", "total"])]
                sheets.append({
                    "name": sheet_name,
                    "columns": cols,
                    "rows": rows,
                    "freeze_header": True,
                    "auto_width": True,
                    "currency_columns": currency_cols,
                    "include_totals_row": False,
                })

        if not has_complex_nested:
            cols = ["Metric / Attribute", "Value"]
            rows = [[str(k).replace("_", " ").title(), v] for k, v in data.items()]
            currency_cols = [c for c in cols if any(kw in c.lower() for kw in ["price", "revenue", "cost", "sales", "amount", "total"])]
            sheets.append({
                "name": "Summary",
                "columns": cols,
                "rows": rows,
                "freeze_header": True,
                "auto_width": True,
                "currency_columns": currency_cols,
                "include_totals_row": False,
            })

    elif isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            cols = [str(k).replace("_", " ").title() for k in data[0].keys()]
            rows = [[item.get(k) for k in data[0].keys()] for item in data]
        elif isinstance(data[0], list):
            cols = [f"Column {i+1}" for i in range(len(data[0]))]
            rows = data
        else:
            cols = ["Items"]
            rows = [[x] for x in data]

        currency_cols = [c for c in cols if any(kw in c.lower() for kw in ["price", "revenue", "cost", "sales", "amount", "total"])]
        sheets.append({
            "name": "Report Data",
            "columns": cols,
            "rows": rows,
            "freeze_header": True,
            "auto_width": True,
            "currency_columns": currency_cols,
            "include_totals_row": len(currency_cols) > 0,
        })

    if not sheets:
        sheets.append({
            "name": "Report",
            "columns": ["Title", "Details"],
            "rows": [[title, "Generated Document Report"]],
            "freeze_header": True,
            "auto_width": True,
        })

    return {
        "format": "xlsx",
        "title": title,
        "filename": title.lower().replace(" ", "_"),
        "sheets": sheets,
        "charts": [],
    }



def _fallback_word_spec(title: str, data: Any) -> dict[str, Any]:
    """Construct a safe, valid default Word specification directly from source data."""
    sections = []

    if isinstance(data, dict):
        table_rows = []
        for k, v in data.items():
            table_rows.append([str(k).replace("_", " ").title(), str(v)])
        
        sections.append({
            "heading": "Summary Overview",
            "heading_level": 1,
            "content": f"Key information for {title}.",
            "table": {
                "headers": ["Property", "Value"],
                "rows": table_rows
            }
        })
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        headers = [str(k).replace("_", " ").title() for k in data[0].keys()]
        rows = [[item.get(k) for k in data[0].keys()] for item in data]
        sections.append({
            "heading": "Detailed Data",
            "heading_level": 1,
            "content": f"The following table details records for {title}.",
            "table": {
                "headers": headers,
                "rows": rows
            }
        })
    else:
        sections.append({
            "heading": "Overview",
            "heading_level": 1,
            "content": str(data) if data else f"This report provides details regarding {title}."
        })

    return {
        "format": "docx",
        "title": title,
        "subtitle": "Vyapar Sakha Business Report",
        "filename": title.lower().replace(" ", "_"),
        "sections": sections
    }


async def build_document_spec_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Build a structured document specification (ExcelDocumentSpec or WordDocumentSpec).
    Uses LLM structured output with fallback to deterministic construction.
    """
    doc_type = state.get("document_type", "excel")
    request = state.get("request", {})
    title = request.get("title") or "Business Report"
    source_data = state.get("source_data") or request.get("data")
    requirements = request.get("requirements", [])
    repair_instructions = state.get("repair_instructions", {})

    llm = get_small_llm()
    spec_dict = None

    if llm:
        try:
            if doc_type == "excel":
                prompt = (
                    f"Create a comprehensive Excel workbook specification for title: '{title}'.\n"
                    f"Requirements: {requirements}\n"
                    f"Repair Instructions: {repair_instructions}\n"
                    f"Source Data: {str(source_data)[:4000]}\n"
                    "Define worksheets, column names, data rows, formatting, and optional charts."
                )
                structured_llm = llm.with_structured_output(ExcelDocumentSpec)
                res = await structured_llm.ainvoke(prompt)
                spec_dict = res.model_dump() if hasattr(res, "model_dump") else res.dict()
            else:
                prompt = (
                    f"Create a clear Word document specification for title: '{title}'.\n"
                    f"Requirements: {requirements}\n"
                    f"Repair Instructions: {repair_instructions}\n"
                    f"Source Data: {str(source_data)[:4000]}\n"
                    "Define document title, sections, headings, paragraphs, lists, and tables."
                )
                structured_llm = llm.with_structured_output(WordDocumentSpec)
                res = await structured_llm.ainvoke(prompt)
                spec_dict = res.model_dump() if hasattr(res, "model_dump") else res.dict()
        except Exception as exc:
            LOGGER.warning("build_document_spec.llm_failed", error=str(exc))

    if not spec_dict:
        # Use deterministic fallback builder
        if doc_type == "excel":
            spec_dict = _fallback_excel_spec(title, source_data)
        else:
            spec_dict = _fallback_word_spec(title, source_data)

    LOGGER.info("build_document_spec.complete", doc_type=doc_type, title=title)
    return {"document_spec": spec_dict}


# ---------------------------------------------------------------------------
# Node 3: Generate Document
# ---------------------------------------------------------------------------

async def generate_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Execute deterministic Python generator (openpyxl or python-docx) based on document_spec.
    """
    doc_type = state.get("document_type", "excel")
    spec = state.get("document_spec", {})

    try:
        if doc_type == "excel":
            file_path, filename = generate_excel_document(spec, TEMP_DOC_DIR)
        else:
            file_path, filename = generate_word_document(spec, TEMP_DOC_DIR)

        LOGGER.info("generate_document.success", file_path=file_path, filename=filename)
        return {
            "generated_file_path": file_path,
            "generated_filename": filename,
        }
    except Exception as exc:
        LOGGER.error("generate_document.failed", error=str(exc), exc_info=True)
        return {
            "generated_file_path": "",
            "generated_filename": "",
            "validation_result": {"valid": False, "errors": [f"Generator crash: {str(exc)}"]},
            "error": f"Failed to generate file: {str(exc)}",
        }


# ---------------------------------------------------------------------------
# Node 4: Validate Document
# ---------------------------------------------------------------------------

async def validate_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Perform deterministic validation on the generated file.
    """
    file_path = state.get("generated_file_path")
    doc_type = state.get("document_type", "excel")
    spec = state.get("document_spec", {})

    if not file_path or not os.path.exists(file_path):
        val_result = {
            "valid": False,
            "errors": ["Generated file path is empty or file does not exist on disk."],
            "warnings": [],
        }
    else:
        if doc_type == "excel":
            val_result = validate_excel_document(file_path, spec)
        else:
            val_result = validate_word_document(file_path, spec)

    errors = val_result.get("errors", [])
    LOGGER.info(
        "validate_document.result",
        valid=val_result.get("valid", False),
        errors=errors,
    )

    return {
        "validation_result": val_result,
        "validation_errors": errors,
    }


# ---------------------------------------------------------------------------
# Node 5: Repair Document
# ---------------------------------------------------------------------------

async def repair_document_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Prepare repair instructions and increment retry_count when validation fails.
    """
    current_retry = state.get("retry_count", 0) + 1
    val_errors = state.get("validation_errors", [])

    LOGGER.info(
        "repair_document.attempt",
        attempt=current_retry,
        max_retries=state.get("max_retries", 2),
        errors=val_errors,
    )

    repair_actions = [f"Fix error: {err}" for err in val_errors]
    repair_instructions = {
        "repair_required": True,
        "errors": val_errors,
        "repair_actions": repair_actions,
    }

    return {
        "retry_count": current_retry,
        "repair_instructions": repair_instructions,
    }


# ---------------------------------------------------------------------------
# Node 6: Upload File
# ---------------------------------------------------------------------------

async def upload_file_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Upload the validated document to Cloudflare R2 persistent storage.
    """
    file_path = state.get("generated_file_path")
    filename = state.get("generated_filename") or "document.bin"
    mime_type = state.get("mime_type", "application/octet-stream")
    store_id = state.get("store_id", "store")
    user_id = state.get("user_id", "user")

    file_id = f"file_{uuid.uuid4().hex[:16]}"
    r2_service = get_r2_storage_service()

    storage_key = r2_service.generate_storage_key(store_id, user_id, file_id, filename)
    upload_res = r2_service.upload_file(file_path, storage_key, mime_type)

    download_url = None
    try:
        download_url = r2_service.generate_presigned_url(storage_key, expires_in=3600)
    except Exception as exc:
        LOGGER.warning("upload_file.presigned_url_failed", error=str(exc))

    file_metadata = {
        "file_id": file_id,
        "filename": filename,
        "mime_type": mime_type,
        "size": upload_res.get("size_bytes", 0),
        "storage_key": storage_key,
        "document_type": state.get("document_type", "excel"),
        "status": "ready",
        "download_url": download_url,
    }

    LOGGER.info("upload_file.complete", file_id=file_id, storage_key=storage_key)

    return {
        "file_id": file_id,
        "storage_key": storage_key,
        "file_metadata": file_metadata,
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Node 7: Create Result
# ---------------------------------------------------------------------------

async def create_result_node(state: DocumentGenerationState) -> dict[str, Any]:
    """
    Clean up temporary local file and build final structured response payload.
    """
    file_path = state.get("generated_file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            LOGGER.info("create_result.temp_cleanup", path=file_path)
        except Exception as exc:
            LOGGER.warning("create_result.temp_cleanup_failed", path=file_path, error=str(exc))

    status = state.get("status")
    val_result = state.get("validation_result", {"valid": False})
    attempts = state.get("retry_count", 0) + 1

    if status == "success" and val_result.get("valid", False):
        final_meta = state.get("file_metadata", {})
        LOGGER.info("create_result.success", file_id=final_meta.get("file_id"))
        return {
            "status": "success",
        }
    else:
        err_msg = state.get("error") or "Document could not be generated after maximum retries."
        LOGGER.warning("create_result.failed", error=err_msg, attempts=attempts)
        return {
            "status": "failed",
            "error": err_msg,
        }
