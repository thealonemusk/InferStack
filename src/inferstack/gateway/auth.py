"""API-key authentication.

Deliberately simple: a static set of bearer tokens from configuration. There is
no user model, no key rotation and no scopes, because the gateway's job here is
to stop an endpoint being open to the internet, not to be an identity provider.

Two details that are easy to get wrong and are worth getting right:

* **Constant-time comparison.** A plain ``==`` on a secret leaks its prefix
  through timing. ``secrets.compare_digest`` does not.
* **No early exit across the key set.** Checking every configured key, rather
  than returning on the first match, keeps the work independent of *which* key
  was supplied.
"""

from __future__ import annotations

import hashlib
import secrets

from inferstack.config import GatewayConfig
from inferstack.gateway.errors import unauthorized

BEARER_PREFIX = "bearer "


def extract_bearer_token(authorization: str | None) -> str | None:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    if not authorization.lower().startswith(BEARER_PREFIX):
        return None
    token = authorization[len(BEARER_PREFIX) :].strip()
    return token or None


def is_valid_key(token: str, api_keys: list[str]) -> bool:
    """Constant-time membership test against the configured keys."""
    # Hashing first gives compare_digest fixed-length inputs, so the comparison
    # cost does not vary with key length either.
    supplied = hashlib.sha256(token.encode()).digest()
    valid = False
    for key in api_keys:
        expected = hashlib.sha256(key.encode()).digest()
        # Bitwise-or rather than `return True`: every key is always checked.
        valid |= secrets.compare_digest(supplied, expected)
    return valid


def authenticate(config: GatewayConfig, authorization: str | None) -> str | None:
    """Authorise a request, returning a non-sensitive key fingerprint.

    The fingerprint identifies *which* key was used in logs and, from Phase 7,
    for per-key rate limiting - without ever writing the key itself to a log.

    Raises:
        GatewayError: 401 when auth is required and the key is missing or wrong.
    """
    if not config.require_auth:
        return None

    if not config.api_keys:
        # Requiring auth with no keys configured would reject every request with
        # a confusing 401. Say what is actually wrong.
        raise unauthorized(
            "Authentication is enabled but no API keys are configured. "
            "Set gateway.api_keys, or disable gateway.require_auth."
        )

    token = extract_bearer_token(authorization)
    if token is None:
        raise unauthorized("Provide an API key as 'Authorization: Bearer <key>'.")

    if not is_valid_key(token, config.api_keys):
        raise unauthorized()

    return key_fingerprint(token)


def key_fingerprint(token: str) -> str:
    """A short, stable, non-reversible id for a key, safe to log."""
    return hashlib.sha256(token.encode()).hexdigest()[:12]
