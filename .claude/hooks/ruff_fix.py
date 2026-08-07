"""PostToolUse hook: ruff lint fixes on the file Claude just wrote.

Reads the hook payload on stdin, does nothing unless a .py file inside this repo
was touched, and never fails the tool call — a formatter that blocks edits is
worse than one that occasionally misses a file.

Two deliberate omissions:

* `ruff format` is NOT run. This codebase is hand-formatted and diverges from
  ruff's formatter in 18 files (config.py's compact `cfg(...)` blocks most
  visibly). Running it per-edit would silently reformat whole files as a side
  effect of a one-line change.
* F401 (unused import) is NOT auto-fixed. A multi-step edit routinely adds an
  import in one call and uses it in the next; deleting it in between turns the
  intermediate state into a NameError.

Both are enforceable deliberately with `ruff format .` / `ruff check --fix .`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():  # POSIX venv layout, or a checkout without one
    PYTHON = REPO / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    raw = (payload.get("tool_response") or {}).get("filePath") or (
        payload.get("tool_input") or {}
    ).get("file_path")
    if not raw:
        return 0

    target = Path(raw)
    if target.suffix != ".py" or not target.is_file():
        return 0

    # Only touch files in this repo — Claude may edit files elsewhere.
    try:
        target.resolve().relative_to(REPO)
    except ValueError:
        return 0

    subprocess.run(
        [
            str(PYTHON), "-m", "ruff", "check", "--fix", "--quiet",
            "--ignore", "F401",
            str(target),
        ],
        capture_output=True,
        timeout=30,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
