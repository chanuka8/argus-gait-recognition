import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from security_layer.auth import (
    AuthenticationInfrastructureError,
    SessionToken,
    get_authenticated_operator,
    get_operator_store,
    get_session_store,
)
from security_layer.authorization import normalize_role
from security_layer.password_hasher import get_password_hasher

logger = logging.getLogger("ARGUS.AuthRouter")

auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, description="Operator username")
    password: str = Field(..., min_length=1, description="Operator password")
    role: str | None = Field(None, description="Optional system role filter")


class LoginResponse(BaseModel):
    success: bool = True
    token: str
    operator: dict[str, Any]
    expires_at: float


class LogoutResponse(BaseModel):
    success: bool = True
    message: str = "Session terminated successfully"


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8)


class ChangePasswordResponse(BaseModel):
    success: bool = True
    message: str = "Password changed successfully"
    token: str


@auth_router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
)
async def login(req: LoginRequest, request: Request):
    """Authenticate an operator and establish a cryptographic session.

    Enforces:
      - Argon2id password verification (with fail-safe rehash-on-login for legacy credentials)
      - Never trusts client-asserted identity or roles
      - Issues an opaque, random session token bound to server-side session state
    """
    try:
        operator_store = get_operator_store()
        user_data, error_msg = operator_store.authenticate_operator(
            username=req.username,
            password=req.password,
            role=req.role,
        )
    except AuthenticationInfrastructureError as exc:
        logger.error(f"[AUTH_INFRASTRUCTURE_FAILURE] {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Contact system administration.",
            headers={"Retry-After": "30"},
        ) from exc

    if error_msg or not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_msg or "Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Resolve client IP
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    session_store = get_session_store()
    session = session_store.create_session(
        operator_id=user_data.get("id", user_data.get("username", req.username)),
        username=user_data.get("username", req.username),
        role=normalize_role(user_data.get("role", "investigator")),
        name=user_data.get("name", ""),
        nic=user_data.get("nic", ""),
        image=user_data.get("image", ""),
        status_val=user_data.get("status", "Active"),
        source_ip=client_ip,
    )

    logger.info(f"[AUTH_SUCCESS] Operator '{session.username}' authenticated ({session.role}) from {client_ip}")

    return LoginResponse(
        success=True,
        token=session.token,
        operator=session.to_profile_dict(),
        expires_at=session.expires_at,
    )


@auth_router.post(
    "/logout",
    response_model=LogoutResponse,
    status_code=status.HTTP_200_OK,
)
async def logout(
    session: Annotated[SessionToken, Depends(get_authenticated_operator)],
):
    """Revoke the current operator session."""
    session_store = get_session_store()
    session_store.revoke_session(session.token)
    logger.info(f"[AUTH_LOGOUT] Session revoked for operator '{session.username}'")
    return LogoutResponse(success=True, message="Session terminated successfully")


@auth_router.get(
    "/me",
    status_code=status.HTTP_200_OK,
)
async def get_current_operator_profile(
    session: Annotated[SessionToken, Depends(get_authenticated_operator)],
):
    """Retrieve verified profile information for the authenticated operator."""
    return session.to_profile_dict()


@auth_router.post(
    "/change-password",
    response_model=ChangePasswordResponse,
    status_code=status.HTTP_200_OK,
)
async def change_password(
    req: ChangePasswordRequest,
    request: Request,
    session: Annotated[SessionToken, Depends(get_authenticated_operator)],
):
    """Change the authenticated operator's password.

    Enforces:
      - Verification of current password
      - Minimum password length (8 chars)
      - Mandatory Argon2id hashing for the new password
      - Invalidation of all existing sessions for this operator
    """
    if len(req.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters long",
        )

    operator_store = get_operator_store()
    user_data, col, doc_id = operator_store.get_operator(session.username)
    if not user_data or not col or not doc_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Operator account record not found",
        )

    # Verify current password
    stored_hash = user_data.get("password_hash") or user_data.get("password")
    hasher = get_password_hasher()
    is_valid, _ = hasher.verify(req.current_password, stored_hash or "")
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password verification failed",
        )

    # Hash new password with Argon2id and update
    success = operator_store.create_or_update_operator(
        collection_name=col,
        doc_id=doc_id,
        username=session.username,
        password=req.new_password,
        role=session.role,
        name=session.name,
        nic=session.nic,
        image=session.image,
        status_val=session.status,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update operator password",
        )

    # Invalidate all existing sessions for security
    session_store = get_session_store()
    session_store.revoke_all_for_operator(session.operator_id)

    # Establish fresh session for the caller
    client_ip = request.client.host if request.client else "unknown"
    new_session = session_store.create_session(
        operator_id=session.operator_id,
        username=session.username,
        role=session.role,
        name=session.name,
        nic=session.nic,
        image=session.image,
        status_val=session.status,
        source_ip=client_ip,
    )

    logger.info(f"[PASSWORD_CHANGED] Operator '{session.username}' changed password successfully.")

    return ChangePasswordResponse(
        success=True,
        message="Password changed successfully",
        token=new_session.token,
    )


class VerifyPasswordRequest(BaseModel):
    password: str = Field(..., min_length=1)


class VerifyPasswordResponse(BaseModel):
    valid: bool


@auth_router.post(
    "/verify-password",
    response_model=VerifyPasswordResponse,
    status_code=status.HTTP_200_OK,
)
async def verify_operator_password(
    req: VerifyPasswordRequest,
    session: Annotated[SessionToken, Depends(get_authenticated_operator)],
):
    """Verify operator password server-side for sensitive administrative actions."""
    op_store = get_operator_store()
    user_data, _, _ = op_store.get_operator(session.username)
    if not user_data:
        raise HTTPException(status_code=404, detail="Operator account not found")

    stored = user_data.get("password_hash") or user_data.get("password")
    if not stored:
        return VerifyPasswordResponse(valid=False)

    is_valid, _ = get_password_hasher().verify(req.password, stored)
    return VerifyPasswordResponse(valid=is_valid)


class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    role: str = Field(..., min_length=3)
    name: str = Field("", min_length=0)
    nic: str = Field("", min_length=0)
    image: str = Field("", min_length=0)


@auth_router.post(
    "/admin/users",
    status_code=status.HTTP_201_CREATED,
)
async def create_operator_account(
    req: CreateUserRequest,
    session: Annotated[SessionToken, Depends(get_authenticated_operator)],
):
    """Admin-only endpoint for provisioning new operators with mandatory Argon2id password hashing."""
    actor_role = session.role.lower()
    if actor_role not in ("root_admin", "root admin", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to create operators",
        )

    # Privilege separation: Admin cannot create Root Admin
    role_clean = req.role.strip().lower()
    if actor_role == "admin" and role_clean in ("root_admin", "root admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators cannot provision Root Administrator accounts",
        )

    op_store = get_operator_store()
    try:
        existing, _, _ = op_store.get_operator(req.username)
    except AuthenticationInfrastructureError as exc:
        logger.error(f"[USER_CREATE_INFRASTRUCTURE_FAILURE] {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Contact system administration.",
        ) from exc

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Operator with this username already exists",
        )

    collection_name = "admins" if role_clean in ("admin", "root_admin", "root admin") else "investigators"

    try:
        success = op_store.create_or_update_operator(
            collection_name=collection_name,
            doc_id=req.username.strip().lower(),
            username=req.username,
            password=req.password,
            role=req.role,
            name=req.name,
            nic=req.nic,
            image=req.image,
        )
    except AuthenticationInfrastructureError as exc:
        logger.error(f"[USER_CREATE_INFRASTRUCTURE_FAILURE] {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Contact system administration.",
        ) from exc

    if not success:
        raise HTTPException(status_code=500, detail="Failed to create operator account")

    logger.info(f"[USER_CREATED] Operator '{req.username}' created by '{session.username}' ({session.role})")
    return {"success": True, "username": req.username, "role": req.role}


class UpdateUserRequest(BaseModel):
    name: str | None = None
    role: str | None = None
    nic: str | None = None
    image: str | None = None
    password: str | None = None
    status: str | None = None


@auth_router.put(
    "/admin/users/{username}",
    status_code=status.HTTP_200_OK,
)
async def update_operator_account(
    username: str,
    req: UpdateUserRequest,
    session: Annotated[SessionToken, Depends(get_authenticated_operator)],
):
    """Admin-only endpoint for updating operator profiles and password re-hashing."""
    actor_role = session.role.lower()
    if actor_role not in ("root_admin", "root admin", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to update operators",
        )

    import time
    op_store = get_operator_store()
    try:
        user_data, col, doc_id = op_store.get_operator(username)
    except AuthenticationInfrastructureError as exc:
        logger.error(f"[USER_UPDATE_INFRASTRUCTURE_FAILURE] {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Contact system administration.",
        ) from exc

    if not user_data or not col or not doc_id:
        raise HTTPException(status_code=404, detail="Operator account not found")

    target_role = (user_data.get("role") or "").lower()

    # Privilege separation: Admin cannot modify Root Admin
    if actor_role == "admin" and target_role in ("root_admin", "root admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators cannot modify Root Administrator accounts",
        )

    # Privilege separation: Admin cannot escalate anyone to Root Admin
    if req.role and actor_role == "admin" and req.role.lower() in ("root_admin", "root admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators cannot escalate accounts to Root Administrator",
        )

    new_role = req.role or user_data.get("role", "investigator")
    new_name = req.name if req.name is not None else user_data.get("name", "")
    new_nic = req.nic if req.nic is not None else user_data.get("nic", "")
    new_image = req.image if req.image is not None else user_data.get("image", "")
    new_status = req.status if req.status is not None else user_data.get("status", "Active")

    try:
        if req.password and req.password.strip():
            success = op_store.create_or_update_operator(
                collection_name=col,
                doc_id=doc_id,
                username=username,
                password=req.password,
                role=new_role,
                name=new_name,
                nic=new_nic,
                image=new_image,
                status_val=new_status,
            )
        else:
            hasher = get_password_hasher()
            stored_hash = user_data.get("password_hash")
            if not stored_hash and user_data.get("password"):
                stored_hash = hasher.hash(user_data["password"])

            data = {
                "name": new_name,
                "username": username.strip().lower(),
                "password_hash": stored_hash,
                "password_migrated": True,
                "role": new_role.lower(),
                "nic": new_nic,
                "image": new_image,
                "status": new_status,
                "lastLogin": user_data.get("lastLogin", "Never"),
                "updated_at": time.time(),
            }
            if op_store.mode == "firebase":
                client = op_store._get_firestore_client()
                if client is None:
                    raise AuthenticationInfrastructureError("Authentication service infrastructure unavailable")
                client.collection(col).document(doc_id).set(data, merge=True)
                success = True
            elif op_store.mode == "offline":
                offline_data = op_store._load_offline_store()
                offline_data.setdefault(col, {})[doc_id] = data
                success = op_store._save_offline_store(offline_data)
            else:
                raise AuthenticationInfrastructureError("Invalid operator store configuration mode")
    except AuthenticationInfrastructureError as exc:
        logger.error(f"[USER_UPDATE_INFRASTRUCTURE_FAILURE] {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Contact system administration.",
        ) from exc

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update operator account")

    logger.info(f"[USER_UPDATED] Operator '{username}' updated by '{session.username}'")
    return {"success": True, "username": username}


@auth_router.delete(
    "/admin/users/{username}",
    status_code=status.HTTP_200_OK,
)
async def delete_operator_account(
    username: str,
    session: Annotated[SessionToken, Depends(get_authenticated_operator)],
):
    """Root-Admin ONLY endpoint for deleting operator accounts."""
    if session.role.lower() not in ("root_admin", "root admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Root Administrator privileges required to delete operators",
        )

    op_store = get_operator_store()
    try:
        user_data, col, doc_id = op_store.get_operator(username)
    except AuthenticationInfrastructureError as exc:
        logger.error(f"[USER_DELETE_INFRASTRUCTURE_FAILURE] {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Contact system administration.",
        ) from exc

    if not user_data or not col or not doc_id:
        raise HTTPException(status_code=404, detail="Operator account not found")

    try:
        op_store.delete_operator(col, doc_id)
    except AuthenticationInfrastructureError as exc:
        logger.error(f"[USER_DELETE_INFRASTRUCTURE_FAILURE] {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Contact system administration.",
        ) from exc

    # Invalidate all active sessions for this deleted user
    get_session_store().revoke_all_for_operator(user_data.get("id", username))
    logger.info(f"[USER_DELETED] Operator '{username}' deleted by '{session.username}'")
    return {"success": True, "message": f"Operator {username} deleted successfully"}
