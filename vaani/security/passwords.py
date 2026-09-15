"""Password hashing.

`hashlib.scrypt`, from the standard library, rather than bcrypt or argon2 behind
passlib. The design document named passlib; this is a deliberate departure, for
the reason the rest of this codebase keeps its base install bare — an air-gapped
government deployment installs what is in the wheel and nothing else, and a
password scheme is the last thing that should depend on a package being
available at install time. scrypt is memory-hard, has been in the standard
library since 3.6, and needs no build step.

The stored form carries its own parameters:

    scrypt$n$r$p$<salt base64>$<derived key base64>

so raising the cost later leaves every existing hash verifiable, and
`needs_rehash` says which rows to upgrade on the owner's next successful login.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCHEME = "scrypt"

# ~16 MiB and roughly 100 ms on the deployment's hardware. High enough to make a
# stolen table expensive, low enough that a login does not stall the event loop
# the live calls are sharing — which is also why callers run it off-thread.
_N = 2**14
_R = 8
_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32

# Long enough to survive a stolen hash, short enough that operators do not write
# it on a monitor. Length is the only rule: composition rules push people toward
# "Password1!" and buy nothing.
MIN_LENGTH = 12


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_KEY_BYTES
    )
    return "$".join([SCHEME, str(_N), str(_R), str(_P), _b64(salt), _b64(derived)])


def verify_password(password: str, stored: str) -> bool:
    """Never raises. A row hand-edited in the database, or written by a scheme
    this version does not know, fails the login rather than the endpoint."""
    try:
        scheme, n, r, p, salt_b64, key_b64 = stored.split("$")
        if scheme != SCHEME:
            return False
        salt = _unb64(salt_b64)
        expected = _unb64(key_b64)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError, MemoryError):
        return False
    # Constant time: a byte-by-byte comparison leaks how much of a guess was
    # right to anyone who can measure the response.
    return hmac.compare_digest(derived, expected)


def needs_rehash(stored: str) -> bool:
    """True when the row was written with weaker parameters than we use now."""
    try:
        scheme, n, r, p, _salt, _key = stored.split("$")
    except ValueError:
        return True
    return scheme != SCHEME or (int(n), int(r), int(p)) != (_N, _R, _P)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))
