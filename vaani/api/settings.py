"""Settings API — the control surface behind the admin settings page.

`GET /schema` returns a self-describing form definition derived from the config
model, so the settings page renders itself and a new setting never needs a
matching UI change.

Secrets are write-only across this boundary: they go in as cleartext and come
back masked, forever. An admin page that can display an API key turns any XSS
into a credential leak, and a support engineer screen-sharing the settings tab
should not be able to read the organisation's keys.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from vaani.core.logging import get_logger
from vaani.providers.base import Message
from vaani.settings_store import SettingsError

log = get_logger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/schema")
async def schema(request: Request) -> dict[str, Any]:
    store = request.app.state.settings_store
    return {"groups": store.schema(), "values": store.public_values()}


@router.get("")
async def values(request: Request) -> dict[str, Any]:
    store = request.app.state.settings_store
    return {
        "values": store.public_values(),
        "overridden": sorted(store.overrides),
    }


@router.put("")
async def update(request: Request, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Validate, persist, and hot-apply a settings change.

    Refused while calls are live: swapping the LLM or speech model underneath a
    caller mid-sentence produces a garbled turn at best. The operator is told to
    wait or end the calls rather than being allowed to break them.
    """
    store = request.app.state.settings_store
    services = request.app.state.services
    manager = request.app.state.calls

    force = bool(patch.pop("_force", False))
    if manager.live_count and not force:
        raise HTTPException(
            409,
            f"{manager.live_count} call(s) in progress. Wait for them to finish, "
            f"or resend with _force to apply anyway.",
        )

    try:
        changed = store.update(patch)
    except SettingsError as exc:
        raise HTTPException(422, {"message": "Invalid settings", "errors": exc.errors}) from exc

    if not changed:
        return {"changed": [], "rebuilt": [], "values": store.public_values()}

    try:
        rebuilt = await services.reload(store.settings)
    except Exception as exc:
        # The new settings are already persisted but a provider would not start
        # — almost always a bad credential. Roll back so the running platform
        # keeps working and the operator sees why.
        log.exception("failed to apply settings; rolling back")
        store.reset(changed)
        raise HTTPException(
            422,
            {
                "message": f"Could not apply those settings: {exc}",
                "errors": {"_": str(exc)},
                "rolled_back": changed,
            },
        ) from exc

    return {"changed": changed, "rebuilt": rebuilt, "values": store.public_values()}


@router.post("/reset")
async def reset(
    request: Request, keys: list[str] | None = Body(None, embed=True)
) -> dict[str, Any]:
    """Drop overrides and fall back to the environment baseline."""
    store = request.app.state.settings_store
    services = request.app.state.services
    removed = store.reset(keys)
    rebuilt = await services.reload(store.settings) if removed else []
    return {"reset": removed, "rebuilt": rebuilt, "values": store.public_values()}


@router.post("/test")
async def test(request: Request, target: str = Body("llm", embed=True)) -> dict[str, Any]:
    """Prove the configured provider actually works, before a caller finds out.

    A wrong API key is invisible until the first turn of the first real call,
    where it costs a caller their time and the operator their credibility. This
    makes it a button.
    """
    services = request.app.state.services
    import time

    started = time.monotonic()
    try:
        if target == "llm":
            completion = await services.llm.complete(
                [
                    Message(role="system", content="Reply with exactly: OK"),
                    Message(role="user", content="Say OK."),
                ],
                max_tokens=16,
            )
            detail = completion.text[:120] or "(empty reply)"
        elif target == "tts":
            audio = await services.tts.synthesize("Connection test.")
            detail = f"{len(audio) / (16000 * 2):.1f}s of audio"
        elif target == "stt":
            # 300 ms of silence: exercises the transport and credentials without
            # asserting anything about recognition quality.
            transcript = await services.stt.transcribe(b"\x00\x00" * 4800)
            detail = f"transcript: {transcript.text!r}"
        elif target == "knowledge":
            hits = await services.retriever.search("connection test")
            detail = f"{len(hits)} passage(s) matched"
        else:
            raise HTTPException(400, f"Unknown test target {target!r}")
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("provider test failed", extra={"target": target, "error": str(exc)})
        return {
            "target": target,
            "ok": False,
            "error": _readable(exc),
            "latency_ms": int((time.monotonic() - started) * 1000),
        }

    return {
        "target": target,
        "ok": True,
        "detail": detail,
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


def _readable(exc: Exception) -> str:
    """Turn a provider exception into something an operator can act on."""
    text = str(exc)
    name = type(exc).__name__
    if "authentication" in name.lower() or "401" in text:
        return "The API key was rejected. Check it is correct and still active."
    if "permission" in name.lower() or "403" in text:
        return "That key does not have access to this model."
    if "notfound" in name.lower() or "404" in text:
        return "The model name was not recognised by the provider."
    if "ratelimit" in name.lower() or "429" in text:
        return "Rate limited by the provider. Try again shortly."
    if "connect" in name.lower() or "timeout" in name.lower():
        return f"Could not reach the provider: {text}"
    return f"{name}: {text}" if text else name
