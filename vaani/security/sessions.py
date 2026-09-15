"""Signed session tokens for the portal.

Stateless and signed rather than a row per session: the deployment is a single
uvicorn worker sharing one event loop with the live calls, and a database read
on every request to the console — which polls — is latency the calls would pay
for.

Stateless costs one thing, and it is worth naming: a token stays valid until it
expires, so revoking access cannot rely on deleting a row. It does not have to.
The guard reloads the account on every request and refuses a suspended one, so
withdrawing access takes effect on the person's next click either way.

    <user id>.<issued at>.<signature>

Signed with HMAC-SHA256 over the first two fields.
"""

from __future__ import annotations

import base64
import hmac
import time
from hashlib import sha256

# Long enough for a working day, short enough that a laptop left open overnight
# is not an open console in the morning.
DEFAULT_TTL_S = 12 * 3600


class SessionSigner:
    def __init__(self, secret: str, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        if not secret:
            raise ValueError("a session signing secret is required")
        self._key = secret.encode("utf-8")
        self._ttl = ttl_s

    def issue(self, user_id: int, *, issued_at: float | None = None) -> str:
        stamp = int(issued_at if issued_at is not None else time.time())
        payload = f"{user_id}.{stamp}"
        return f"{payload}.{self._sign(payload)}"

    def verify(self, token: str | None) -> int | None:
        """The user id, or None for anything that is not a live, untampered token."""
        if not token:
            return None
        try:
            user_id, stamp, signature = token.split(".")
            issued = int(stamp)
            expected = self._sign(f"{user_id}.{stamp}")
            # Constant time: a byte-by-byte comparison leaks how much of a
            # forged signature was right to anyone who can measure the response.
            if not hmac.compare_digest(signature, expected):
                return None
            if issued + self._ttl < time.time():
                return None
            return int(user_id)
        except (ValueError, TypeError):
            return None

    def _sign(self, payload: str) -> str:
        digest = hmac.new(self._key, payload.encode("utf-8"), sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
