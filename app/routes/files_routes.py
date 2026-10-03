"""
app/routes/files_routes.py
==========================
FastAPI routes for listing and downloading agent-generated files
(Excel, Word, CSV, PDF, etc.) stored in Cloudflare R2.

Endpoints:
    GET  /ai/{store_id}/files                               — list all files for a store
    GET  /ai/{store_id}/files/download?key={storage_key}   — generate a presigned download URL
"""

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.r2_storage_service import get_r2_storage_service

router = APIRouter(tags=["agent_files"])

LOGGER = structlog.get_logger("vyaparsathi.ai.files_routes")

# Allowed extensions for agent-generated documents
DOCUMENT_EXTENSIONS = {
    ".xlsx", ".xls", ".docx", ".doc", ".csv", ".pdf", ".txt", ".json"
}

FILE_TYPE_LABELS = {
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".docx": "Word",
    ".doc": "Word",
    ".csv": "CSV",
    ".pdf": "PDF",
    ".txt": "Text",
    ".json": "JSON",
}

FILE_TYPE_ICONS = {
    ".xlsx": "📊",
    ".xls": "📊",
    ".docx": "📝",
    ".doc": "📝",
    ".csv": "📋",
    ".pdf": "📄",
    ".txt": "📃",
    ".json": "🗂️",
}


def _ext(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx != -1 else ""


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _parse_object(obj: dict) -> dict | None:
    key: str = obj.get("Key", "")
    filename = key.rsplit("/", 1)[-1] if "/" in key else key
    ext = _ext(filename)

    if ext not in DOCUMENT_EXTENSIONS:
        return None

    size_bytes: int = obj.get("Size", 0)
    last_modified = obj.get("LastModified")

    return {
        "storage_key": key,
        "filename": filename,
        "file_type": FILE_TYPE_LABELS.get(ext, "File"),
        "icon": FILE_TYPE_ICONS.get(ext, "📁"),
        "extension": ext,
        "size_bytes": size_bytes,
        "size_label": _format_size(size_bytes),
        "last_modified": last_modified.isoformat() if last_modified else None,
    }


@router.get("/{store_id}/files")
async def list_agent_files(store_id: str):
    """List all agent-generated document files in R2 for a store, newest first."""
    try:
        svc = get_r2_storage_service()
        client = svc._get_client()

        prefix = f"stores/{store_id}/"
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=svc.bucket_name, Prefix=prefix)

        files = []
        for page in pages:
            for obj in page.get("Contents", []):
                parsed = _parse_object(obj)
                if parsed:
                    files.append(parsed)

        files.sort(key=lambda f: f["last_modified"] or "", reverse=True)

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "files": files,
                    "total": len(files),
                    "store_id": store_id,
                },
            }
        )

    except ValueError as exc:
        LOGGER.warning("files_routes.list.r2_not_configured", error=str(exc))
        raise HTTPException(status_code=503, detail="File storage is not configured.")
    except Exception as exc:
        LOGGER.error("files_routes.list.error", store_id=store_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to list agent files.")


@router.get("/{store_id}/files/download")
async def get_file_download_url(
    store_id: str,
    key: str = Query(..., description="R2 storage key of the file"),
    expires_in: int = Query(3600, ge=60, le=86400, description="Presigned URL TTL in seconds"),
):
    """Generate a temporary presigned download URL for a file in R2."""
    expected_prefix = f"stores/{store_id}/"
    if not key.startswith(expected_prefix):
        raise HTTPException(status_code=403, detail="Access denied: key does not belong to this store.")

    filename = key.rsplit("/", 1)[-1] if "/" in key else key
    ext = _ext(filename)
    if ext not in DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported for download.")

    try:
        svc = get_r2_storage_service()
        presigned_url = svc.generate_presigned_url(key, expires_in=expires_in)

        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "download_url": presigned_url,
                    "filename": filename,
                    "expires_in": expires_in,
                },
            }
        )

    except ValueError as exc:
        LOGGER.warning("files_routes.download.r2_not_configured", error=str(exc))
        raise HTTPException(status_code=503, detail="File storage is not configured.")
    except Exception as exc:
        LOGGER.error("files_routes.download.error", store_id=store_id, key=key, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to generate download URL.")
