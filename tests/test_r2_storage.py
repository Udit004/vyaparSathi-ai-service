from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.config.settings import Settings
from app.services.r2_storage import (
    R2AccessDeniedError,
    R2ConfigurationError,
    R2ObjectNotFoundError,
    R2StorageService,
    build_document_object_key,
)


def r2_settings(**overrides: str | None) -> Settings:
    values = {
        "cloudflare_account_id": "account-123",
        "cloudflare_r2_endpoint": "https://account-123.r2.cloudflarestorage.com",
        "cloudflare_r2_access_key_id": "test-access-key",
        "cloudflare_r2_secret_access_key": "test-secret-key",
        "cloudflare_r2_bucket_name": "documents",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "test failure"}}, "test")


def test_settings_load_r2_configuration() -> None:
    settings = r2_settings()
    assert settings.cloudflare_r2_bucket_name == "documents"
    assert settings.cloudflare_r2_endpoint.endswith(".r2.cloudflarestorage.com")


def test_client_initialization_uses_r2_configuration() -> None:
    with patch("app.services.r2_storage.boto3.client") as create_client:
        R2StorageService(settings=r2_settings())

    _, kwargs = create_client.call_args
    assert kwargs["endpoint_url"] == "https://account-123.r2.cloudflarestorage.com"
    assert kwargs["region_name"] == "auto"
    assert kwargs["aws_access_key_id"] == "test-access-key"
    assert kwargs["aws_secret_access_key"] == "test-secret-key"


def test_missing_configuration_fails_clearly() -> None:
    with pytest.raises(R2ConfigurationError, match="CLOUDFLARE_R2_BUCKET_NAME"):
        R2StorageService(settings=r2_settings(cloudflare_r2_bucket_name=None), client=MagicMock())


@pytest.mark.asyncio
async def test_upload_download_metadata_exists_and_delete() -> None:
    client = MagicMock()
    client.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"report"))}
    client.head_object.return_value = {"ContentLength": 6, "ContentType": "application/pdf"}
    storage = R2StorageService(settings=r2_settings(), client=client)

    await storage.upload_bytes(b"report", "stores/store/users/user/files/file/report.pdf", "application/pdf")
    assert await storage.download_bytes("stores/store/users/user/files/file/report.pdf") == b"report"
    assert await storage.file_exists("stores/store/users/user/files/file/report.pdf") is True
    assert (await storage.get_file_metadata("stores/store/users/user/files/file/report.pdf"))["ContentLength"] == 6
    await storage.delete_file("stores/store/users/user/files/file/report.pdf")

    assert client.put_object.call_args.kwargs["ContentType"] == "application/pdf"
    client.delete_object.assert_called_once()


@pytest.mark.asyncio
async def test_missing_object_returns_false_for_exists_and_raises_for_download() -> None:
    client = MagicMock()
    client.head_object.side_effect = client_error("NoSuchKey")
    client.get_object.side_effect = client_error("NoSuchKey")
    storage = R2StorageService(settings=r2_settings(), client=client)

    assert await storage.file_exists("missing.pdf") is False
    with pytest.raises(R2ObjectNotFoundError):
        await storage.download_bytes("missing.pdf")


@pytest.mark.asyncio
async def test_access_denied_is_mapped_to_storage_exception() -> None:
    client = MagicMock()
    client.put_object.side_effect = client_error("AccessDenied")
    storage = R2StorageService(settings=r2_settings(), client=client)

    with pytest.raises(R2AccessDeniedError):
        await storage.upload_bytes(b"report", "report.pdf")


@pytest.mark.asyncio
async def test_generates_presigned_download_url() -> None:
    client = MagicMock()
    client.generate_presigned_url.return_value = "https://example.invalid/signed-url"
    storage = R2StorageService(settings=r2_settings(), client=client)

    assert await storage.generate_presigned_download_url("report.pdf", expires_in=300) == "https://example.invalid/signed-url"
    assert client.generate_presigned_url.call_args.kwargs["ExpiresIn"] == 300


def test_document_object_key_is_scoped_and_file_id_is_backend_generated() -> None:
    file_id, object_key = build_document_object_key(
        store_id="store123", user_id="user456", filename="monthly-sales.xlsx"
    )
    assert file_id.startswith("file_")
    assert object_key == f"stores/store123/users/user456/files/{file_id}/monthly-sales.xlsx"

    with pytest.raises(ValueError, match="filename"):
        build_document_object_key(store_id="store123", user_id="user456", filename="../secret.pdf")
    with pytest.raises(ValueError, match="filename"):
        build_document_object_key(store_id="store123", user_id="user456", filename="..\\secret.pdf")
