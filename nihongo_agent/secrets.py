"""Secure secret resolution via Google Cloud Secret Manager with environment fallback.

No API keys, database passwords, or service credentials are ever hardcoded in
the repository. Runtime components resolve sensitive configuration through
``get_secret()``, which queries ``SecretManagerServiceClient`` when a cloud
project is configured and falls back cleanly to environment variables during
local development and offline evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

try:
    from google.cloud import secretmanager  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional when running without cloud extras
    secretmanager = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SecretResolutionResult:
    """Metadata describing how a configuration secret was resolved without exposing its value."""

    secret_id: str
    source: str  # "gcp_secret_manager", "environment", or "default_fallback"
    value: str


class SecretResolver:
    """Resolves secrets from Google Cloud Secret Manager with safe offline fallback."""

    def __init__(self, project_id: Optional[str] = None) -> None:
        self.project_id = (
            project_id
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
        )
        self._client: Optional[object] = None

    def _get_client(self) -> Optional[object]:
        """Lazily instantiate ``SecretManagerServiceClient`` only when needed."""
        if self._client is not None:
            return self._client
        if secretmanager is None or not self.project_id:
            return None
        if os.environ.get("NIHONGO_AGENT_FORCE_OFFLINE", "").lower() in {"1", "true", "yes"}:
            return None
        try:
            self._client = secretmanager.SecretManagerServiceClient()
            return self._client
        except Exception:
            return None

    def resolve(
        self,
        secret_id: str,
        version: str = "latest",
        default: Optional[str] = None,
    ) -> SecretResolutionResult:
        """Fetch ``secret_id`` from Secret Manager, falling back to environment variables.

        Args:
            secret_id: Identifier of the secret in Secret Manager or environment variable name.
            version: Secret version to access (defaults to ``"latest"``).
            default: Non-sensitive fallback value used during offline testing when unset.

        Returns:
            ``SecretResolutionResult`` containing the resolved value and its provenance.

        Raises:
            RuntimeError: If the secret is absent in Secret Manager, environment, and ``default``.
        """
        if not secret_id or not secret_id.strip():
            raise ValueError("secret_id must be a non-empty string.")

        normalized_id = secret_id.strip()

        # 1. Attempt Google Cloud Secret Manager when a project ID and client are available.
        client = self._get_client()
        if client is not None and self.project_id:
            secret_path = (
                f"projects/{self.project_id}/secrets/{normalized_id}/versions/{version}"
            )
            try:
                response = client.access_secret_version(request={"name": secret_path})  # type: ignore[attr-defined]
                payload = response.payload.data.decode("utf-8").strip()
                if payload:
                    return SecretResolutionResult(
                        secret_id=normalized_id,
                        source="gcp_secret_manager",
                        value=payload,
                    )
            except Exception:
                # Fall through to environment variable resolution when offline or unauthenticated.
                pass

        # 2. Check environment variables (exact match or uppercase snake_case match).
        env_candidates = (normalized_id, normalized_id.upper().replace("-", "_"))
        for env_key in env_candidates:
            env_val = os.environ.get(env_key)
            if env_val is not None and env_val.strip():
                return SecretResolutionResult(
                    secret_id=normalized_id,
                    source="environment",
                    value=env_val.strip(),
                )

        # 3. Use caller-provided safe default for offline evaluation if supplied.
        if default is not None:
            return SecretResolutionResult(
                secret_id=normalized_id,
                source="default_fallback",
                value=default,
            )

        raise RuntimeError(
            f"Secret '{normalized_id}' could not be resolved from Google Cloud Secret Manager "
            f"or environment variables {env_candidates}."
        )


@lru_cache(maxsize=32)
def get_secret(
    secret_id: str,
    project_id: Optional[str] = None,
    version: str = "latest",
    default: Optional[str] = None,
) -> str:
    """Resolve and cache a secret string from Secret Manager or environment variables.

    Args:
        secret_id: Secret identifier in Google Cloud Secret Manager or environment variable name.
        project_id: Optional GCP project ID override.
        version: Secret Manager version tag (default ``"latest"``).
        default: Optional safe fallback string for offline execution.

    Returns:
        Resolved secret string.
    """
    resolver = SecretResolver(project_id=project_id)
    return resolver.resolve(secret_id=secret_id, version=version, default=default).value
