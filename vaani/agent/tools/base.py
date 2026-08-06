"""Tool definitions and the registry.

The design rule this enforces: the LLM never *is* the business logic, it only
decides which business operation to invoke. Booking an appointment, raising a
complaint, checking a bill — each is a plain async Python function with a typed
schema. The model picks one; deterministic code does the work and talks to the
CRM, ERP or SQL Server behind it.

That separation is what makes the platform auditable. Every tool invocation is
logged with its arguments and result, so an operator can reconstruct exactly what
the agent did on any call.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from vaani.core.logging import get_logger

log = get_logger(__name__)

ToolFn = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class ToolContext:
    """Everything a tool may need about the call it is running inside."""

    call_id: str
    caller_number: str | None = None
    agent_key: str = "default"
    language: str = "en"
    # Scratch space shared across tools within one call, e.g. a verified
    # consumer number established by an earlier tool.
    state: dict[str, Any] = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """What goes back to the model.

    `content` is read by the LLM, so it must be short, factual prose — not JSON
    the model has to parse. `control` is read by the pipeline, and is how a tool
    causes something to happen to the call itself (transfer, hang up).
    """

    content: str
    ok: bool = True
    control: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: ToolFn
    # Tools that change the world need a confirmed caller identity first.
    requires_verification: bool = False
    timeout_s: float = 8.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        requires_verification: bool = False,
        timeout_s: float = 8.0,
    ) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    fn=fn,
                    requires_verification=requires_verification,
                    timeout_s=timeout_s,
                )
            )
            return fn

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """Wire schemas for the subset this agent profile is allowed to use.

        Exposing every tool to every agent both confuses small models and is a
        privilege escalation waiting to happen.
        """
        names = allowed if allowed is not None else self.names()
        return [self._tools[n].to_wire() for n in names if n in self._tools]

    async def invoke(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            # Small models hallucinate tool names. Tell the model, don't crash.
            log.warning("unknown tool", extra={"tool": name})
            return ToolResult(
                content=f"There is no tool called {name}. Available: {', '.join(self.names())}",
                ok=False,
            )

        if tool.requires_verification and not ctx.state.get("verified"):
            return ToolResult(
                content=(
                    "The caller has not been verified yet. Ask for and verify their "
                    "registered mobile number before using this tool."
                ),
                ok=False,
            )

        try:
            result = await asyncio.wait_for(
                _call(tool.fn, arguments, ctx), timeout=tool.timeout_s
            )
        except asyncio.TimeoutError:
            log.warning("tool timeout", extra={"tool": name, "timeout_s": tool.timeout_s})
            return ToolResult(
                content="That system did not respond in time. Tell the caller you will "
                "try again or offer to transfer them.",
                ok=False,
            )
        except Exception as exc:  # a broken integration must not drop the call
            log.exception("tool failed", extra={"tool": name})
            return ToolResult(
                content=f"That operation failed: {exc}. Apologise and offer an alternative.",
                ok=False,
            )

        if isinstance(result, ToolResult):
            return result
        return ToolResult(content=str(result))


async def _call(fn: ToolFn, arguments: dict[str, Any], ctx: ToolContext) -> Any:
    """Pass `ctx` only to tools that declare it, so simple tools stay simple."""
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in arguments.items() if k in sig.parameters}
    if "ctx" in sig.parameters:
        kwargs["ctx"] = ctx
    return await fn(**kwargs)
