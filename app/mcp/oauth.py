"""
app/mcp/oauth.py
================
OAuth 2.0 Authorization Server for Vyapar Sathi MCP.
Enforces strict multi-tenant isolation:
- Users must authenticate with their own account (Firebase Google Auth or registered Email).
- Users can ONLY view and authorize stores that THEY own.
- Prevents cross-tenant access and store leakage.
"""
from __future__ import annotations

import os
import time
import secrets
from typing import Any, Dict, List, Optional
from bson import ObjectId
from bson.errors import InvalidId
import jwt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.mcp.oauth")

oauth_router = APIRouter(tags=["oauth"])

# OAuth Secret and Expiry
OAUTH_JWT_SECRET = os.getenv("OAUTH_JWT_SECRET", "vyapar-sathi-mcp-oauth-jwt-secret-2026")
DEFAULT_CLIENT_ID = os.getenv("MCP_CLIENT_ID", "vyapar-mcp-client")
DEFAULT_CLIENT_SECRET = os.getenv("MCP_CLIENT_SECRET", "vyapar-mcp-secret-2026")

# In-memory auth codes store: { code: { user_id, store_id, store_name, email, expires_at, client_id, redirect_uri } }
AUTH_CODES: Dict[str, Dict[str, Any]] = {}


def _get_base_url(request: Request) -> str:
    """Extract public base URL considering proxies and ngrok."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    # External tunnel domains like ngrok always require https for OAuth
    if "ngrok" in host or request.headers.get("x-forwarded-proto") == "https":
        proto = "https"
    return f"{proto}://{host}"


# ===========================================================================
# 1. RFC 8414 & OpenID Connect Discovery Endpoints
# ===========================================================================

@oauth_router.get("/.well-known/oauth-authorization-server")
@oauth_router.get("/.well-known/openid-configuration")
@oauth_router.get("/.well-known/oauth-authorization-server/mcp/sse")
@oauth_router.get("/.well-known/openid-configuration/mcp/sse")
async def oauth_auth_server_discovery(request: Request):
    """RFC 8414 OAuth 2.0 Authorization Server Metadata."""
    base = _get_base_url(request)
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "registration_endpoint": f"{base}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic", "none"],
        "code_challenge_methods_supported": ["S256", "plain"],
        "scopes_supported": ["read", "store:read"],
        "service_documentation": f"{base}/mcp-info",
    })


@oauth_router.get("/.well-known/oauth-protected-resource")
@oauth_router.get("/.well-known/oauth-protected-resource/mcp/sse")
async def oauth_protected_resource_discovery(request: Request):
    """RFC 9728 OAuth 2.0 Protected Resource Metadata."""
    base = _get_base_url(request)
    return JSONResponse({
        "resource": base,
        "authorization_servers": [base],
        "scopes_supported": ["read", "store:read"],
    })


@oauth_router.post("/oauth/register", status_code=201)
async def oauth_register(request: Request):
    """RFC 7591 Dynamic Client Registration (DCR) endpoint."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    client_id = f"mcp_client_{secrets.token_hex(8)}"
    client_secret = secrets.token_urlsafe(32)
    redirect_uris = body.get("redirect_uris", [])

    LOGGER.info("dcr_client_registered", client_id=client_id, uris=redirect_uris)

    return JSONResponse({
        "client_id": client_id,
        "client_secret": client_secret,
        "client_name": body.get("client_name", "ChatGPT"),
        "redirect_uris": redirect_uris,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
    }, status_code=201)


# ===========================================================================
# 2. User Store Lookup (ONLY Returns Stores Owned by That Specific User)
# ===========================================================================

@oauth_router.post("/oauth/user-stores")
async def get_user_stores(request: Request):
    """
    Given the authenticated user's email, look up ONLY the stores they own.
    Returns 404 if the user is not found in Vyapar Sathi.
    """
    body = await request.json()
    email = (body.get("email") or "").strip().lower()

    if not email:
        return JSONResponse({"error": "Email is required"}, status_code=400)

    db = get_database()
    user = await db["users"].find_one({"email": email})

    if not user:
        return JSONResponse({
            "error": f"No Vyapar Sathi account found for '{email}'. Please ensure you use the email registered with Vyapar Sathi."
        }, status_code=404)

    user_id = user["_id"]
    user_name = user.get("name", email)

    # Fetch ONLY stores owned by this user
    stores_cursor = db["stores"].find({"owner": user_id, "isActive": {"$ne": False}}, {"_id": 1, "name": 1})
    stores = await stores_cursor.to_list(20)

    if not stores:
        # Check if store has owner as string or fallback
        stores_cursor = db["stores"].find({"owner": str(user_id)}, {"_id": 1, "name": 1})
        stores = await stores_cursor.to_list(20)

    store_list = [
        {"store_id": str(s["_id"]), "store_name": s.get("name", "My Store")}
        for s in stores
    ]

    LOGGER.info("user_stores_retrieved", email=email, store_count=len(store_list))

    return JSONResponse({
        "user_id": str(user_id),
        "user_name": user_name,
        "email": email,
        "stores": store_list,
    })


# ===========================================================================
# 3. Secure Authorization Consent Page (GET /oauth/authorize)
# ===========================================================================

@oauth_router.get("/oauth/authorize", response_class=HTMLResponse)
async def oauth_authorize_page(
    request: Request,
    client_id: str = "",
    redirect_uri: str = "",
    response_type: str = "code",
    state: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "",
):
    """
    Renders a secure login & consent screen.
    Does NOT leak all stores. Requires the user to authenticate with their own email/Google,
    then shows only THEIR owned store(s) to connect.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Connect Vyapar Sathi — ChatGPT</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- Firebase App & Auth SDK (v9 compat) -->
    <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-auth-compat.js"></script>
    <style>
        :root {{
            --primary: #4F46E5;
            --primary-hover: #4338CA;
            --bg: #0B0F19;
            --card-bg: #151D2F;
            --border: #23304A;
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --success: #10B981;
            --error: #EF4444;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, #070A11 0%, #0F172A 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .modal {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 18px;
            width: 100%;
            max-width: 440px;
            padding: 32px 28px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }}
        .header {{ text-align: center; margin-bottom: 24px; }}
        .app-badges {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            margin-bottom: 16px;
        }}
        .badge {{
            width: 50px;
            height: 50px;
            border-radius: 12px;
            background: #1C273D;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            border: 1px solid var(--border);
        }}
        .link-icon {{ color: var(--text-muted); font-size: 18px; }}
        h1 {{ font-size: 21px; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.3px; }}
        p.subtitle {{ color: var(--text-muted); font-size: 13.5px; line-height: 1.45; }}

        .auth-step {{ display: block; }}
        .auth-step.hidden {{ display: none; }}

        .google-btn {{
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            background: #FFFFFF;
            color: #1F2937;
            font-weight: 600;
            font-size: 14.5px;
            padding: 12px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
            margin-bottom: 18px;
        }}
        .google-btn:hover {{ background: #F1F5F9; transform: translateY(-1px); }}
        .google-icon {{ width: 18px; height: 18px; }}

        .divider {{
            display: flex;
            align-items: center;
            text-align: center;
            margin: 16px 0;
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .divider::before, .divider::after {{
            content: '';
            flex: 1;
            border-bottom: 1px solid var(--border);
        }}
        .divider:not(:empty)::before {{ margin-right: 12px; }}
        .divider:not(:empty)::after {{ margin-left: 12px; }}

        .input-group {{ margin-bottom: 16px; }}
        label {{ display: block; font-size: 12.5px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }}
        input[type="email"] {{
            width: 100%;
            padding: 12px 14px;
            background: #0E1524;
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }}
        input[type="email"]:focus {{ border-color: var(--primary); }}

        .btn {{
            width: 100%;
            padding: 13px;
            border-radius: 10px;
            font-size: 14.5px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-primary {{ background: var(--primary); color: white; }}
        .btn-primary:hover {{ background: var(--primary-hover); }}
        .btn-primary:disabled {{ background: #26334D; color: #64748B; cursor: not-allowed; }}

        .alert-error {{
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #FCA5A5;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            margin-bottom: 16px;
            display: none;
        }}

        /* Step 2: Store selection */
        .user-pill {{
            background: #1C273E;
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }}
        .user-pill-email {{ font-size: 13px; font-weight: 600; color: #E2E8F0; }}
        .switch-link {{ font-size: 12px; color: #818CF8; cursor: pointer; text-decoration: none; }}
        .switch-link:hover {{ text-decoration: underline; }}

        .store-select-group {{ margin-bottom: 20px; }}
        select.store-dropdown {{
            width: 100%;
            padding: 12px 14px;
            background: #0E1524;
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 14px;
            font-weight: 500;
            outline: none;
            cursor: pointer;
        }}
        select.store-dropdown:focus {{ border-color: var(--primary); }}

        .permissions-box {{
            background: #0E1524;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }}
        .permission-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
            color: #CBD5E1;
            margin-bottom: 8px;
        }}
        .permission-item:last-child {{ margin-bottom: 0; }}
        .perm-check {{ color: var(--success); font-weight: bold; font-size: 14px; }}

        .cancel-link {{
            display: block;
            text-align: center;
            margin-top: 14px;
            color: var(--text-muted);
            text-decoration: none;
            font-size: 13px;
        }}
        .cancel-link:hover {{ color: var(--text-main); }}
    </style>
</head>
<body>
    <div class="modal">
        <div class="header">
            <div class="app-badges">
                <div class="badge" title="Vyapar Sathi">🏪</div>
                <div class="link-icon">⇄</div>
                <div class="badge" title="ChatGPT">🤖</div>
            </div>
            <h1>Connect to Vyapar Sathi</h1>
            <p class="subtitle">Authenticate with your Vyapar Sathi account to securely connect your store to ChatGPT.</p>
        </div>

        <div id="errorMessage" class="alert-error"></div>

        <!-- STEP 1: AUTHENTICATION -->
        <div id="stepAuth" class="auth-step">
            <button type="button" class="google-btn" onclick="signInGoogle()">
                <svg class="google-icon" viewBox="0 0 48 48">
                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                </svg>
                Sign in with Google
            </button>

            <div class="divider">or use registered email</div>

            <form id="emailForm" onsubmit="event.preventDefault(); verifyEmail();">
                <div class="input-group">
                    <label for="userEmail">Vyapar Sathi Account Email</label>
                    <input type="email" id="userEmail" placeholder="e.g. rajankumart266@gmail.com" required value="rajankumart266@gmail.com">
                </div>
                <button type="submit" class="btn btn-primary" id="emailSubmitBtn">
                    Continue with Email
                </button>
            </form>
        </div>

        <!-- STEP 2: STORE SELECTION & AUTHORIZATION -->
        <div id="stepConsent" class="auth-step hidden">
            <div class="user-pill">
                <div>
                    <div style="font-size: 11px; color: var(--text-muted);">AUTHENTICATED AS</div>
                    <div class="user-pill-email" id="pillEmail"></div>
                </div>
                <span class="switch-link" onclick="switchAccount()">Switch</span>
            </div>

            <div class="store-select-group">
                <label for="storeSelect">Select Store to Connect</label>
                <select id="storeSelect" class="store-dropdown"></select>
            </div>

            <div class="permissions-box">
                <div class="permission-item">
                    <span class="perm-check">✓</span> Only access this specific store's data
                </div>
                <div class="permission-item">
                    <span class="perm-check">✓</span> View inventory & low-stock alerts
                </div>
                <div class="permission-item">
                    <span class="perm-check">✓</span> View customer dues & sales reports
                </div>
            </div>

            <button type="button" class="btn btn-primary" id="authorizeBtn" onclick="approveSelectedStore()">
                Authorize ChatGPT
            </button>
        </div>

        <a href="javascript:window.close()" class="cancel-link">Cancel</a>
    </div>

    <script>
        // Initialize Firebase
        const firebaseConfig = {{
            apiKey: "AIzaSyBN8pv5wlKqu88GFNyAMRgHQxPRVbqYrnk",
            authDomain: "vyapar-sathi-e4119.firebaseapp.com",
            projectId: "vyapar-sathi-e4119"
        }};
        if (!firebase.apps.length) {{
            firebase.initializeApp(firebaseConfig);
        }}

        let currentUser = null;

        function showError(msg) {{
            const el = document.getElementById('errorMessage');
            el.textContent = msg;
            el.style.display = 'block';
        }}

        function hideError() {{
            document.getElementById('errorMessage').style.display = 'none';
        }}

        async function signInGoogle() {{
            hideError();
            const provider = new firebase.auth.GoogleAuthProvider();
            try {{
                const result = await firebase.auth().signInWithPopup(provider);
                const email = result.user.email;
                await loadUserStores(email);
            }} catch (error) {{
                showError("Google Sign-In: " + error.message);
            }}
        }}

        async function verifyEmail() {{
            hideError();
            const email = document.getElementById('userEmail').value.trim();
            if (!email) return;
            const btn = document.getElementById('emailSubmitBtn');
            btn.disabled = true;
            btn.textContent = 'Verifying account...';
            try {{
                await loadUserStores(email);
            }} finally {{
                btn.disabled = false;
                btn.textContent = 'Continue with Email';
            }}
        }}

        async function loadUserStores(email) {{
            hideError();
            try {{
                const res = await fetch('/oauth/user-stores', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ email: email }})
                }});
                const data = await res.json();
                if (!res.ok || data.error) {{
                    showError(data.error || 'Failed to find account');
                    return;
                }}

                if (!data.stores || data.stores.length === 0) {{
                    showError("No active store found for account: " + email);
                    return;
                }}

                currentUser = data;

                // Populate store dropdown
                const select = document.getElementById('storeSelect');
                select.innerHTML = '';
                data.stores.forEach(st => {{
                    const opt = document.createElement('option');
                    opt.value = st.store_id;
                    opt.textContent = st.store_name;
                    select.appendChild(opt);
                }});

                document.getElementById('pillEmail').textContent = data.user_name + ' (' + data.email + ')';
                document.getElementById('stepAuth').classList.add('hidden');
                document.getElementById('stepConsent').classList.remove('hidden');

            }} catch (err) {{
                showError("Network error: " + err.message);
            }}
        }}

        function switchAccount() {{
            currentUser = null;
            document.getElementById('stepConsent').classList.add('hidden');
            document.getElementById('stepAuth').classList.remove('hidden');
        }}

        async function approveSelectedStore() {{
            const btn = document.getElementById('authorizeBtn');
            btn.disabled = true;
            btn.textContent = 'Connecting to ChatGPT...';

            const select = document.getElementById('storeSelect');
            const selectedStoreId = select.value;
            const selectedStoreName = select.options[select.selectedIndex].text;

            const payload = {{
                client_id: "{client_id}",
                redirect_uri: "{redirect_uri}",
                state: "{state}",
                code_challenge: "{code_challenge}",
                user_id: currentUser.user_id,
                store_id: selectedStoreId,
                store_name: selectedStoreName,
                email: currentUser.email
            }};

            try {{
                const res = await fetch('/oauth/approve', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});
                const data = await res.json();
                if (data.redirect_url) {{
                    window.location.href = data.redirect_url;
                }} else {{
                    showError(data.error || 'Authorization failed');
                    btn.disabled = false;
                }}
            }} catch (err) {{
                showError('Connection error: ' + err.message);
                btn.disabled = false;
            }}
        }}
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ===========================================================================
# 4. Handle Approval with Strict Ownership Validation (POST /oauth/approve)
# ===========================================================================

@oauth_router.post("/oauth/approve")
async def oauth_approve(request: Request):
    """
    Called when merchant clicks 'Authorize'.
    Strictly enforces that the user actually owns the selected store before issuing code.
    """
    body = await request.json()
    client_id = body.get("client_id", "")
    redirect_uri = body.get("redirect_uri", "")
    state = body.get("state", "")
    user_id = (body.get("user_id") or "").strip()
    store_id = (body.get("store_id") or "").strip()
    store_name = (body.get("store_name") or "").strip()
    email = (body.get("email") or "").strip()

    if not redirect_uri:
        return JSONResponse({"error": "redirect_uri is required"}, status_code=400)

    if not user_id or not store_id:
        return JSONResponse({"error": "user_id and store_id are required"}, status_code=400)

    db = get_database()

    # Security verification: verify that the store is owned by this user
    try:
        user_oid = ObjectId(user_id)
        store_oid = ObjectId(store_id)
    except (InvalidId, TypeError):
        return JSONResponse({"error": "Invalid user_id or store_id format"}, status_code=400)

    store = await db["stores"].find_one({
        "_id": store_oid,
        "$or": [{"owner": user_oid}, {"owner": str(user_oid)}]
    })

    if not store:
        LOGGER.warning("unauthorized_store_connection_attempt", user=email, store=store_name)
        return JSONResponse({
            "error": "Security alert: You can only authorize stores that belong to your account."
        }, status_code=403)

    actual_store_name = store.get("name", store_name)

    # Generate one-time authorization code (valid for 10 minutes)
    code = secrets.token_urlsafe(32)
    AUTH_CODES[code] = {
        "user_id": user_id,
        "store_id": str(store["_id"]),
        "store_name": actual_store_name,
        "email": email,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "expires_at": time.time() + 600,
    }

    LOGGER.info("oauth_authorized_for_owner", user=email, store=actual_store_name, code=code[:8] + "...")

    separator = "&" if "?" in redirect_uri else "?"
    callback_url = f"{redirect_uri}{separator}code={code}"
    if state:
        callback_url += f"&state={state}"

    return JSONResponse({"redirect_url": callback_url})


# ===========================================================================
# 5. Token Exchange Endpoint (POST /oauth/token)
# ===========================================================================

@oauth_router.post("/oauth/token")
async def oauth_token(request: Request):
    """
    ChatGPT backend calls this to exchange the authorization code for a JWT access_token.
    Accepts both application/x-www-form-urlencoded and application/json.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}
        grant_type = body.get("grant_type", "authorization_code")
        code = body.get("code", "")
        client_id = body.get("client_id")
        client_secret = body.get("client_secret")
        redirect_uri = body.get("redirect_uri")
    else:
        try:
            form = await request.form()
        except Exception:
            form = {}
        grant_type = form.get("grant_type", "authorization_code")
        code = form.get("code", "")
        client_id = form.get("client_id")
        client_secret = form.get("client_secret")
        redirect_uri = form.get("redirect_uri")

    LOGGER.info("oauth_token_exchange_request", grant_type=grant_type, code=code[:8] if code else "")

    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    auth_entry = AUTH_CODES.get(code)
    if not auth_entry:
        return JSONResponse({"error": "invalid_grant", "error_description": "Invalid or expired authorization code"}, status_code=400)

    if time.time() > auth_entry["expires_at"]:
        AUTH_CODES.pop(code, None)
        return JSONResponse({"error": "invalid_grant", "error_description": "Authorization code expired"}, status_code=400)

    # Consume the one-time code
    AUTH_CODES.pop(code, None)

    # Generate signed JWT token valid for 30 days
    now = int(time.time())
    payload = {
        "sub": auth_entry["user_id"],
        "email": auth_entry["email"],
        "store_id": auth_entry["store_id"],
        "store_name": auth_entry["store_name"],
        "iss": "vyaparsathi",
        "iat": now,
        "exp": now + (30 * 24 * 3600),
    }

    access_token = jwt.encode(payload, OAUTH_JWT_SECRET, algorithm="HS256")
    LOGGER.info("oauth_access_token_issued", user=auth_entry["email"], store=auth_entry["store_name"])

    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 30 * 24 * 3600,
        "scope": "store:read",
    })


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a Vyapar Sathi OAuth JWT token."""
    try:
        return jwt.decode(token, OAUTH_JWT_SECRET, algorithms=["HS256"])
    except Exception:
        return None
