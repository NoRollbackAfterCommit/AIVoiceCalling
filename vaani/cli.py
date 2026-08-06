"""Command line entry points."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="vaani", description="AI voice agent platform")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the platform")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")

    ingest = sub.add_parser("ingest", help="Index documents into the knowledge base")
    ingest.add_argument("paths", nargs="+", help="Files or directories")
    ingest.add_argument("--agent", default="default")

    chat = sub.add_parser("chat", help="Talk to an agent in the terminal (no audio)")
    chat.add_argument("--agent", default="default")

    args = parser.parse_args()
    if args.command == "serve":
        return _serve(args)
    if args.command == "ingest":
        return asyncio.run(_ingest(args))
    if args.command == "chat":
        return asyncio.run(_chat(args))
    return 1


def _serve(args) -> int:
    import uvicorn

    from vaani.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "vaani.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_config=None,  # our JSON formatter owns logging
    )
    return 0


async def _ingest(args) -> int:
    from vaani.config import get_settings
    from vaani.core.registry import build_services

    services = build_services(get_settings())
    await services.retriever.start()

    paths: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            paths.append(path)
        else:
            print(f"skipping missing path: {path}", file=sys.stderr)

    if not paths:
        print("nothing to ingest", file=sys.stderr)
        return 1

    count = await services.retriever.index_paths(paths, agent_key=args.agent)
    print(f"indexed {count} chunks from {len(paths)} file(s) into agent {args.agent!r}")
    return 0


async def _chat(args) -> int:
    """The fastest feedback loop for prompt and knowledge work."""
    from vaani.agent.runtime import ConversationAgent
    from vaani.agent.tools.base import ToolContext
    from vaani.config import get_settings
    from vaani.core.registry import build_services

    services = build_services(get_settings())
    await services.llm.start()
    await services.retriever.start()

    profile = services.profile(args.agent)
    ctx = ToolContext(call_id="cli", agent_key=args.agent, services=services.as_tool_services())
    agent = ConversationAgent(profile, services.llm, services.tools, ctx)

    print(f"\n  {profile.name}: {profile.greeting}\n  (ctrl-c to end)\n")
    try:
        while True:
            try:
                text = input("  you: ").strip()
            except EOFError:
                break
            if not text:
                continue
            turn = await agent.respond(text)
            if turn.tool_calls:
                names = ", ".join(t["name"] for t in turn.tool_calls)
                print(f"       [tools: {names}]")
            print(f"  {profile.name}: {turn.text}   ({turn.latency_ms} ms)\n")
            if turn.control.get("action") in ("hangup", "transfer"):
                print(f"  [call {turn.control['action']}]")
                break
    except KeyboardInterrupt:
        pass
    finally:
        await services.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
