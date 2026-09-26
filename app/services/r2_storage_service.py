"""
app/services/r2_storage_service.py
==================================
Cloudflare R2 Storage Service wrapper using boto3 S3 interface.

Handles persistent file storage and presigned URL generation for generated documents.
"""

import os
from typing import Any
import structlog
import boto3
from botocore.config import Config

from app.config.settings import get_settings

LOGGER = structlog.get_logger("vyaparsathi.ai.services.r2_storage")


class R2StorageService:
    def __init__(self):
        settings = get_settings()
        self.bucket_name = settings.cloudflare_r2_bucket_name
        self.endpoint_url = settings.cloudflare_r2_endpoint
        self.access_key_id = settings.cloudflare_r2_access_key_id
        self.secret_access_key = settings.cloudflare_r2_secret_access_key

        self._s3_client = None

    def _get_client(self):
        if self._s3_client is None:
            if not self.endpoint_url or not self.access_key_id or not self.secret_access_key:
                LOGGER.warning("r2_storage.missing_credentials")
                raise ValueError("Cloudflare R2 credentials or endpoint URL not configured.")
            
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
                region_name="us-east-1",
            )

        return self._s3_client

    def generate_storage_key(self, store_id: str, user_id: str, file_id: str, filename: str) -> str:
        """
        Generate a safe, structured object key in R2.
        Example: stores/store_123/users/user_456/docs/file_789_report.xlsx
        """
        safe_store_id = store_id or "default_store"
        safe_user_id = user_id or "default_user"
        safe_filename = os.path.basename(filename).replace(" ", "_")
        return f"stores/{safe_store_id}/users/{safe_user_id}/docs/{file_id}_{safe_filename}"

    def upload_file(self, local_file_path: str, storage_key: str, mime_type: str) -> dict[str, Any]:
        """
        Upload a local file to Cloudflare R2 bucket.
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file does not exist: {local_file_path}")

        file_size = os.path.getsize(local_file_path)
        client = self._get_client()

        LOGGER.info(
            "r2_storage.uploading",
            local_path=local_file_path,
            storage_key=storage_key,
            bucket=self.bucket_name,
            size=file_size,
        )

        client.upload_file(
            Filename=local_file_path,
            Bucket=self.bucket_name,
            Key=storage_key,
            ExtraArgs={
                "ContentType": mime_type,
            },
        )

        LOGGER.info("r2_storage.upload_success", storage_key=storage_key)

        return {
            "bucket": self.bucket_name,
            "storage_key": storage_key,
            "size_bytes": file_size,
            "mime_type": mime_type,
        }

    def generate_presigned_url(self, storage_key: str, expires_in: int = 3600) -> str:
        """
        Generate a temporary GET presigned URL for downloading a file from R2.
        """
        client = self._get_client()
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": storage_key},
            ExpiresIn=expires_in,
        )
        return url


_R2_SERVICE_INSTANCE = None


def get_r2_storage_service() -> R2StorageService:
    global _R2_SERVICE_INSTANCE
    if _R2_SERVICE_INSTANCE is None:
        _R2_SERVICE_INSTANCE = R2StorageService()
    return _R2_SERVICE_INSTANCE
