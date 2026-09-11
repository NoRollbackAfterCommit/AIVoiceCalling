"""A size ceiling for the endpoints that authenticate themselves.

`/api/telephony/announce` and `/api/telephony/smartflo/handshake` are exempt
from the shared bearer guard (see vaani/api/auth.py), so a stranger reaches
both with no credential at all. Both parse their body before deciding anything,
and Starlette's `Request.body()` concatenates the whole stream with no cap: a
POST declaring two gigabytes became two gigabytes of resident memory on a VM
that shares its RAM with a government website and has no `mem_limit`, plus a
second of joining and parsing on the single event loop every live call is paced
from. The ceiling has to be applied before the first byte is read, which means
it can only be read off the headers.

64 KiB is a hundredfold headroom over either caller — Asterisk's CURL() posts
four short fields, Smartflo's dynamic endpoint a handful — so a carrier that
adds a field still gets through, while a flood costs a few megabytes rather
than the host.
"""

from __future__ import annotations

from fastapi import HTTPException
from starlette.requests import Request

from vaani.core.logging import get_logger

log = get_logger(__name__)

MAX_WEBHOOK_BODY_BYTES = 64 * 1024


def enforce_body_limit(request: Request) -> None:
    """Refuse before anything reads the body. Raises HTTPException, never returns
    a value, so a call site cannot forget to act on the answer.

    A body of undeclared length is refused too: chunked transfer means the size
    is only known once it has all arrived, which is precisely the read this
    exists to prevent. Neither caller sends chunked — both post a short, fully
    buffered body — so 411 is a diagnosis rather than a lost call.
    """
    declared = request.headers.get("content-length")
    if declared is None:
        # No content-length and no transfer-encoding means no body at all under
        # HTTP/1.1, so there is nothing to read and nothing to refuse.
        if "chunked" not in request.headers.get("transfer-encoding", "").lower():
            return
        log.warning("refused a body of undeclared length", extra={"route": request.url.path})
        raise HTTPException(411, "a Content-Length is required on this endpoint")

    try:
        length = int(declared)
    except ValueError:
        log.warning("refused an unreadable Content-Length", extra={"route": request.url.path})
        raise HTTPException(411, "a Content-Length is required on this endpoint") from None

    if length < 0:
        # `int("-1")` parses happily and `-1 > ceiling` is False, so a negative
        # declared length skipped the size check and the unbounded read went
        # ahead. Every server in front rejects one today; the guard in the
        # process that does the reading should not depend on that.
        log.warning("refused a negative Content-Length", extra={"route": request.url.path})
        raise HTTPException(411, "a Content-Length is required on this endpoint")

    if length > MAX_WEBHOOK_BODY_BYTES:
        log.warning(
            "refused an oversized request body",
            extra={
                "route": request.url.path,
                "declared": length,
                "ceiling": MAX_WEBHOOK_BODY_BYTES,
            },
        )
        raise HTTPException(413, "request body is too large")
