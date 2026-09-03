# ARGUS AI — Firebase Security & Credential Governance

## 1. Zero-Trust Credential Governance

ARGUS AI follows a strict zero-trust credential isolation policy.

### 1.1 Backend Service Account Security
- **Backend Only**: Firebase Admin SDK service account keys (`config/firebase-service-account.json`) are strictly restricted to the server-side Python environment.
- **Client Prohibition**: Service account JSON files, private keys, client emails, and internal service credentials are **never** bundled, imported, or transmitted to frontend browser clients.
- **Resolution Order**:
  1. `FIREBASE_SERVICE_ACCOUNT_PATH` environment variable.
  2. `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
  3. Safe default repository path: `config/firebase-service-account.json`.
  4. If no credentials exist or validation fails: safe closed fallback to offline mode.

### 1.2 Git Tracking Protection
All service account files are explicitly ignored in `.gitignore`:
```gitignore
# Firebase & Cloud Credentials
config/firebase-service-account.json
config/*service-account*.json
firebase-service-account.json
*.service-account.json
```
**Verification Evidence**:
Running `git check-ignore -v config/firebase-service-account.json` confirms:
```
.gitignore:50:config/firebase-service-account.json    config/firebase-service-account.json
```

---

## 2. Zero-Secret Logging Policy

1. **No Token Logging**: Plaintext passwords, password hashes, Firebase ID tokens, session tokens, and RTSP credentials are never logged at any log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
2. **Safe Health Diagnostics**: The `check_connection_health()` method sanitizes all diagnostic output, exposing only:
   - `mode`: `"live"` | `"offline"`
   - `status`: `"HEALTHY"` | `"UNHEALTHY"`
   - `credential`: `"FOUND"` | `"MISSING"` | `"INVALID"`
   - `firestore`: `"CONNECTED"` | `"DISCONNECTED"`
   - `storage`: `"CONNECTED"` | `"DISCONNECTED"`
   - `project_id`: `"argus-17702"`
   - `latency_ms`: Float
   - `retry_queue_size`: Integer

---

## 3. Authoritative Authentication Architecture

ARGUS AI does **NOT** use Firebase Authentication ID tokens (JWTs issued by Google Identity Toolkit).

Instead, ARGUS AI uses **Custom Server-Side SessionToken Authentication** backed by Firestore operator records:

```
[Browser Client]
   │
   ├─ 1. POST /api/v1/auth/login {username, password, role}
   │        │
   │        ▼
   │     [FastAPI Backend: security_layer/auth.py]
   │        │
   │        ├─ 2. Query Firestore collection 'admins' or 'investigators' (via Firebase Admin SDK)
   │        ├─ 3. Verify Argon2id password hash ($argon2id$v=19$m=65536,t=3,p=4$...)
   │        ├─ 4. Create cryptographically unguessable SessionToken in SessionStore
   │        └─ 5. Return HTTP 200 {token: "<session_token>", operator: {...}}
   │
   ├─ 6. Store token in sessionStorage('argus_session_token')
   │
   ├─ 7. Subsequent API calls (cameras, events, metrics, streaming):
   │     fetch(url, { headers: { 'Authorization': 'Bearer ' + session_token } })
   │
   └─ 8. Backend dependency get_current_operator_session(request) validates session token.
```

### Prohibited Query-Parameter Authentication
Authentication tokens **MUST NOT** be passed in URL query strings (e.g. `?token=...` or `?access_token=...`). `extract_bearer_token(request)` strictly inspects `Authorization: Bearer <session_token>`. Any request relying on query parameters is rejected with `HTTP 401 Unauthorized`.

---

## 4. Firebase Admin SDK Security Boundary & Rules Bypass

> [!IMPORTANT]
> **Firebase Admin SDK operations bypass Firestore and Storage Security Rules entirely.**
> 
> The Google Cloud Firebase Admin SDK operates with full administrative service-account privileges (`roles/firebase.admin`). As a consequence:
> 1. Client-side `firestore.rules` and `storage.rules` **do not apply** to backend operations executed through the Python Firebase Admin SDK.
> 2. The **authoritative security boundary** for all ARGUS operations is the backend FastAPI authentication and RBAC layer (`security_layer/auth.py`).
> 3. `firestore.rules` and `storage.rules` serve strictly as defense-in-depth against direct, unauthorized browser-side SDK access to cloud resources.

---

## 5. Role-Based Access Control (RBAC)

The system defines 3 principal operational roles:

| Role | Permissions | Endpoints Allowed | Endpoints Denied |
| :--- | :--- | :--- | :--- |
| **Root Admin** | Full administrative rights, operator provisioning, system config | All (`/cameras/start`, `/cameras/stop`, `/auth/operators`, `/model/*`) | None |
| **Admin** | Camera deployment, subject enrollment, model promotion, audit inspection | Camera start/stop, enrollment, model promotion, audits | Super-admin provisioning |
| **Investigator** | Surveillance monitoring, live stream viewing, event query, identification | `/cameras` (view), `/cameras/{id}/stream`, `/events`, `/metrics`, `/identify/*` | `/cameras/start`, `/cameras/stop`, `/auth/operators`, model promotion |

---

## 6. Camera & Stream Authentication

### 6.1 Control Plane
- `POST /api/v1/cameras/start` and `POST /api/v1/cameras/stop` require an authenticated administrative session token (`Authorization: Bearer <session_token>`).
- Unauthenticated requests are rejected with `HTTP 401 Unauthorized`.
- Non-admin sessions (e.g. `investigator`) are rejected with `HTTP 403 Forbidden`.

### 6.2 Data Plane (Live Video Feeds)
- `GET /api/v1/cameras/{id}/stream` and `GET /api/v1/cameras/{id}/snapshot` enforce session token validation.
- Native browser `<img>` tags cannot attach custom `Authorization` headers. To resolve this securely without exposing tokens in URL query strings:
  1. The frontend `LiveCameraFeed` component issues an authenticated `fetch` with `getAuthHeaders()`.
  2. The response stream body is parsed for multipart JPEG boundaries (`0xFF, 0xD8` to `0xFF, 0xD9`).
  3. Frame data is rendered in-memory via `URL.createObjectURL(blob)` and revoked on each subsequent frame.
  4. Fallback: Authenticated snapshot polling via `getCameraSnapshot(cameraId)`.
- Query parameters like `?token=` are strictly rejected with `HTTP 401 Unauthorized`.

---

## 7. Security Rules Verification

Production rules are maintained in:
- `firestore.rules`: Restricts reading of biometric embeddings, model registry, and audit logs to authenticated roles; prevents client-side writes to audit trails.
- `storage.rules`: Enforces maximum file size limits (50MB evidence, 500MB models), restricts file MIME types, and disallows anonymous upload/download.
