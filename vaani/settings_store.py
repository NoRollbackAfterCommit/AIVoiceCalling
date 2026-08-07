"""Runtime-mutable settings, layered over the environment.

Three layers, lowest priority first:

    1. field defaults in vaani.config
    2. environment variables / .env      — the deployment baseline
    3. data/settings.json                — overrides written by the admin portal

Layer 3 exists because an operator configuring an agent should not need shell
access to the container. Layer 2 still wins at *first boot* of a fresh install,
which is what makes a Kubernetes deployment reproducible: the manifest sets the
baseline, and anything an operator later changes in the UI is an explicit,
persisted, auditable override rather than a silent drift.

Secrets live in the same file but never leave this process in cleartext. The API
returns a masked hint (`sk-ant-…4f2a`) so the UI can show that a key is present
without being able to read it back — a settings page that echoes API keys is one
XSS away from leaking every credential the platform holds.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vaani.config import Settings
from vaani.core.logging import get_logger

log = get_logger(__name__)

MASK = "••••••••"


class SettingsError(ValueError):
    """A rejected settings update. Carries per-field messages for the UI."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


class SettingsStore:
    def __init__(self, path: str | Path = "./data/settings.json") -> None:
        self._path = Path(path)
        self._overrides: dict[str, Any] = self._read()
        self._settings = self._build()

    # -- access -------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def overrides(self) -> dict[str, Any]:
        return dict(self._overrides)

    def _build(self) -> Settings:
        # Environment first (pydantic-settings reads it), then overrides on top.
        try:
            return Settings(**self._overrides)
        except ValidationError:
            # A settings file written by an older version can carry a field this
            # build no longer accepts. Dropping the bad keys keeps the service
            # bootable, which matters more than honouring a stale override.
            log.exception("settings file rejected; falling back to environment")
            valid = {k: v for k, v in self._overrides.items() if k in Settings.model_fields}
            try:
                return Settings(**valid)
            except ValidationError:
                log.error("overrides unusable, using environment only")
                self._overrides = {}
                return Settings()

    # -- persistence --------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            log.exception("could not read settings file", extra={"path": str(self._path)})
            return {}

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._overrides, indent=2, sort_keys=True), encoding="utf-8")
        # The file holds API keys. Restrict it before it lands at its final name,
        # so there is no window where it is readable by other local accounts.
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:  # Windows and some mounted volumes ignore chmod
            pass
        tmp.replace(self._path)

    # -- mutation -----------------------------------------------------------

    def update(self, patch: dict[str, Any]) -> list[str]:
        """Validate and persist a partial update. Returns the changed keys.

        Validation runs against the *merged* result rather than the patch alone,
        so a change that is only invalid in combination with existing settings is
        still caught. Nothing is written unless the whole patch validates.
        """
        unknown = [k for k in patch if k not in Settings.model_fields]
        if unknown:
            raise SettingsError({k: "unknown setting" for k in unknown})

        secrets = Settings.secret_fields()
        candidate = dict(self._overrides)
        changed: list[str] = []

        for key, value in patch.items():
            # The UI round-trips the mask for untouched secret fields. Treat it
            # as "leave this alone" — writing the literal bullets would silently
            # destroy a working API key.
            if key in secrets and isinstance(value, str) and set(value.strip()) <= {"•"}:
                continue
            if isinstance(value, str) and not value.strip() and _nullable(key):
                value = None
            if candidate.get(key) != value:
                candidate[key] = value
                changed.append(key)

        if not changed:
            return []

        try:
            merged = Settings(**candidate)
        except ValidationError as exc:
            raise SettingsError(
                {
                    ".".join(str(p) for p in err["loc"]) or "settings": err["msg"]
                    for err in exc.errors()
                }
            ) from exc

        self._overrides = candidate
        self._settings = merged
        self._write()
        log.info("settings updated", extra={"changed": changed})
        return changed

    def reset(self, keys: list[str] | None = None) -> list[str]:
        """Drop overrides, reverting those settings to the environment baseline."""
        targets = keys if keys is not None else list(self._overrides)
        removed = [k for k in targets if k in self._overrides]
        for key in removed:
            del self._overrides[key]
        if removed:
            self._settings = self._build()
            self._write()
            log.info("settings reset", extra={"reset_keys": removed})
        return removed

    # -- presentation -------------------------------------------------------

    def public_values(self) -> dict[str, Any]:
        """Current effective values, with every secret masked."""
        secrets = Settings.secret_fields()
        out: dict[str, Any] = {}
        for name in Settings.model_fields:
            value = getattr(self._settings, name)
            out[name] = mask_secret(value) if name in secrets else value
        return out

    def schema(self) -> list[dict[str, Any]]:
        """UI descriptor: every field, grouped, in declaration order."""
        groups: dict[str, dict[str, Any]] = {}
        for name, field in Settings.model_fields.items():
            meta = Settings.field_meta(name)
            group = meta.get("group") or "Other"
            groups.setdefault(group, {"group": group, "fields": []})
            groups[group]["fields"].append(
                {
                    "key": name,
                    "label": meta.get("label") or name.replace("_", " ").capitalize(),
                    "help": meta.get("help") or "",
                    "type": _ui_type(field.annotation, meta),
                    "options": meta.get("options"),
                    "secret": bool(meta.get("secret")),
                    "depends_on": meta.get("depends_on"),
                    "restart": meta.get("restart", True),
                    "overridden": name in self._overrides,
                    "default": _jsonable(field.default),
                }
            )
        return list(groups.values())


def mask_secret(value: Any) -> str | None:
    """Show that a credential exists, and its last four characters, nothing more."""
    if value in (None, ""):
        return None
    text = str(value)
    return f"{MASK}{text[-4:]}" if len(text) > 4 else MASK


def _nullable(key: str) -> bool:
    field = Settings.model_fields[key]
    return field.default is None


def _ui_type(annotation: Any, meta: dict[str, Any]) -> str:
    if meta.get("secret"):
        return "secret"
    if meta.get("options"):
        return "select"
    text = str(annotation)
    if "bool" in text:
        return "bool"
    if "float" in text:
        return "float"
    if "int" in text:
        return "int"
    return "string"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
