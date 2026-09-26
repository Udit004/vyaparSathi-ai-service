# Cloudflare R2 Storage

`app.services.r2_storage.R2StorageService` is the backend-only storage primitive
for future generated documents. It is intentionally independent from FastAPI
routes, the LangGraph agent, and document-generation code.

## R2 token setup

1. In Cloudflare, create an R2 API token with `Object Read & Write` permission.
2. Restrict that token to the one bucket used by this service.
3. Copy its access-key ID and secret access key into the AI service `.env` file.
4. Never place either value in a frontend environment file or send it through an
   agent prompt, API response, or LangGraph state.

## Required environment variables

```dotenv
CLOUDFLARE_ACCOUNT_ID=<account-id>
CLOUDFLARE_R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
CLOUDFLARE_R2_ACCESS_KEY_ID=<r2-access-key-id>
CLOUDFLARE_R2_SECRET_ACCESS_KEY=<r2-secret-access-key>
CLOUDFLARE_R2_BUCKET_NAME=<bucket-name>
```

## Usage

```python
from app.services.r2_storage import R2StorageService

storage = R2StorageService()
file_id, object_key = storage.build_object_key(
    store_id=store_id,
    user_id=user_id,
    filename="monthly-sales.xlsx",
)
await storage.upload_bytes(content, object_key, content_type=mime_type)
download_url = await storage.generate_presigned_download_url(object_key)
```

Object keys are backend-generated and scoped as
`stores/{store_id}/users/{user_id}/files/{file_id}/{filename}`. The filename is
only descriptive; `file_id` is the unique identifier.

## Connection check and tests

Run the mocked unit tests without R2 credentials:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_r2_storage.py
```

With the required `.env` values configured, a trusted backend process can call
`await R2StorageService().health_check()` to verify access with `HeadBucket`.
It does not create, modify, or delete data.
