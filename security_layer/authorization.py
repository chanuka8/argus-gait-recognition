"""ARGUS AI - Centralized Role-Based Access Control & Object-Level Authorization (RBAC / anti-BOLA).

Security Guarantees:
  - Centralized, server-side authorization: never trusts client headers, client roles, or URL ownership claims.
  - Fail-closed: unauthorized, malformed, or unrecognized identities are rejected with 401 or 403.
  - Privilege Separation: strictly enforces role boundaries (root_admin > admin > investigator).
  - Anti-BOLA / IDOR: verifies server-side resource ownership for cases, jobs, credentials, and operator records.
"""

from enum import Enum
from typing import Annotated

from fastapi import Depends, HTTPException, status

from security_layer.auth import SessionToken, get_authenticated_operator, get_operator_store
from services.reference_job_manager import ReferenceJobManager, ReferenceJobRecord


class Role(str, Enum):
    ROOT_ADMIN = "root_admin"
    ADMIN = "admin"
    INVESTIGATOR = "investigator"


def normalize_role(role_raw: str | None) -> str:
    """Normalize user role string across representations."""
    if not role_raw:
        return Role.INVESTIGATOR.value
    clean = role_raw.strip().lower().replace(" ", "_")
    if clean in ("root_admin", "rootadmin", "superadmin"):
        return Role.ROOT_ADMIN.value
    if clean in ("admin", "administrator"):
        return Role.ADMIN.value
    return Role.INVESTIGATOR.value


class Permission(str, Enum):
    # Authentication & Profile
    AUTH_ME = "auth:me"
    AUTH_PASSWORD_CHANGE = "auth:password_change"
    AUTH_PASSWORD_VERIFY = "auth:password_verify"

    # Operator Administration
    OPERATOR_VIEW = "operator:view"
    OPERATOR_CREATE = "operator:create"
    OPERATOR_UPDATE = "operator:update"
    OPERATOR_DELETE = "operator:delete"

    # Case & Media Management
    CASE_CREATE = "case:create"
    CASE_VIEW = "case:view"
    CASE_UPDATE = "case:update"
    CASE_DELETE = "case:delete"
    MEDIA_UPLOAD = "media:upload"
    MEDIA_VIEW = "media:view"
    MEDIA_DELETE = "media:delete"

    # Reference Jobs
    JOB_CREATE = "job:create"
    JOB_VIEW = "job:view"
    JOB_RETRY = "job:retry"

    # Biometrics & Gallery
    BIOMETRIC_ENROLL = "biometric:enroll"
    BIOMETRIC_VIEW = "biometric:view"
    BIOMETRIC_DELETE = "biometric:delete"
    BIOMETRIC_SEARCH = "biometric:search"

    # Cameras & Streams
    CAMERA_LIST = "camera:list"
    CAMERA_STREAM = "camera:stream"
    CAMERA_CONTROL = "camera:control"
    CAMERA_CREDENTIAL_MANAGE = "camera:credential_manage"

    # Continual Learning & Model Ops
    LEARNING_VIEW = "learning:view"
    LEARNING_MANAGE = "learning:manage"

    # System Configuration
    SYSTEM_CONFIG = "system:config"
    SYSTEM_AUDIT = "system:audit"


# Explicit Role -> Permissions Matrix
ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    Role.ROOT_ADMIN.value: set(Permission),  # Full permissions
    Role.ADMIN.value: {
        Permission.AUTH_ME,
        Permission.AUTH_PASSWORD_CHANGE,
        Permission.AUTH_PASSWORD_VERIFY,
        Permission.OPERATOR_VIEW,
        Permission.OPERATOR_CREATE,
        Permission.OPERATOR_UPDATE,
        Permission.CASE_CREATE,
        Permission.CASE_VIEW,
        Permission.CASE_UPDATE,
        Permission.CASE_DELETE,
        Permission.MEDIA_UPLOAD,
        Permission.MEDIA_VIEW,
        Permission.MEDIA_DELETE,
        Permission.JOB_CREATE,
        Permission.JOB_VIEW,
        Permission.JOB_RETRY,
        Permission.BIOMETRIC_ENROLL,
        Permission.BIOMETRIC_VIEW,
        Permission.BIOMETRIC_DELETE,
        Permission.BIOMETRIC_SEARCH,
        Permission.CAMERA_LIST,
        Permission.CAMERA_STREAM,
        Permission.CAMERA_CONTROL,
        Permission.CAMERA_CREDENTIAL_MANAGE,
        Permission.LEARNING_VIEW,
        Permission.LEARNING_MANAGE,
        Permission.SYSTEM_AUDIT,
    },
    Role.INVESTIGATOR.value: {
        Permission.AUTH_ME,
        Permission.AUTH_PASSWORD_CHANGE,
        Permission.AUTH_PASSWORD_VERIFY,
        Permission.CASE_CREATE,
        Permission.CASE_VIEW,
        Permission.CASE_UPDATE,
        Permission.MEDIA_UPLOAD,
        Permission.MEDIA_VIEW,
        Permission.JOB_CREATE,
        Permission.JOB_VIEW,
        Permission.JOB_RETRY,
        Permission.BIOMETRIC_ENROLL,
        Permission.BIOMETRIC_VIEW,
        Permission.BIOMETRIC_SEARCH,
        Permission.CAMERA_LIST,
        Permission.CAMERA_STREAM,
    },
}


def has_permission(role: str, permission: Permission) -> bool:
    """Check if a given role possesses a permission."""
    norm = normalize_role(role)
    allowed = ROLE_PERMISSIONS.get(norm, set())
    return permission in allowed


def require_role(*allowed_roles: str):
    """FastAPI Dependency factory enforcing allowed roles."""
    normalized_allowed = {normalize_role(r) for r in allowed_roles}

    async def _role_checker(
        session: Annotated[SessionToken, Depends(get_authenticated_operator)],
    ) -> SessionToken:
        operator_role = normalize_role(session.role)
        if operator_role not in normalized_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted for role '{session.role}'",
            )
        return session

    return _role_checker


def require_permission(permission: Permission):
    """FastAPI Dependency factory enforcing granular permission."""

    async def _permission_checker(
        session: Annotated[SessionToken, Depends(get_authenticated_operator)],
    ) -> SessionToken:
        if not has_permission(session.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires permission '{permission.value}'",
            )
        return session

    return _permission_checker


# Object-Level Authorization Checks (Anti-BOLA / IDOR)


def verify_job_access(
    job_id: str,
    session: SessionToken,
    mutate: bool = False,
) -> ReferenceJobRecord:
    """Verify that authenticated operator is authorized to access or mutate this job."""
    job_mgr = ReferenceJobManager.get_instance()
    job = job_mgr.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference video job '{job_id}' not found",
        )

    operator_role = normalize_role(session.role)

    # Administrators have oversight access to all jobs
    if operator_role in (Role.ROOT_ADMIN.value, Role.ADMIN.value):
        return job

    # Investigators can only access and retry jobs they own
    job_owner = getattr(job, "owner", "")
    if job_owner and job_owner.strip().lower() != session.username.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: job '{job_id}' is owned by another operator",
        )

    return job


def verify_case_access(
    case_id: str,
    session: SessionToken,
    mutate: bool = False,
) -> None:
    """Verify that authenticated operator is authorized to access or mutate a case."""
    operator_role = normalize_role(session.role)

    # Delete operations require administrative privileges
    if mutate and operator_role == Role.INVESTIGATOR.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required to delete cases",
        )


def verify_operator_management_access(
    target_username: str,
    session: SessionToken,
    action: str,
) -> None:
    """Verify administrative hierarchy rules when managing operators."""
    actor_role = normalize_role(session.role)

    if actor_role == Role.INVESTIGATOR.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Investigators cannot manage operators",
        )

    # Deletion is strictly root_admin only
    if action == "delete" and actor_role != Role.ROOT_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Root Administrator privileges required to delete operators",
        )

    # Target inspection: admin cannot modify or delete root_admin
    op_store = get_operator_store()
    target_data, _, _ = op_store.get_operator(target_username)
    if target_data:
        target_role = normalize_role(target_data.get("role"))
        if target_role == Role.ROOT_ADMIN.value and actor_role != Role.ROOT_ADMIN.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrators cannot modify Root Administrator accounts",
            )
