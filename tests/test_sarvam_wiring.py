"""Sarvam must be reachable through config alone — that is the whole point of
the registry. Also guards the reload() key lists, which are silently easy to
forget: a missing key means the admin portal saves a setting that never takes
effect on the running provider.
"""

from __future__ import annotations

import inspect

import pytest

from vaani.config import Settings
from vaani.core.registry import Services, build_stt, build_tts


def test_sarvam_is_a_selectable_provider():
    s = Settings(stt_provider="sarvam", tts_provider="sarvam", sarvam_api_key="k")
    assert build_stt(s).name == "sarvam"
    assert build_tts(s).name == "sarvam"


def test_defaults_still_select_mocks():
    """A bare install with no keys must keep booting."""
    s = Settings()
    assert build_stt(s).name == "mock"
    assert build_tts(s).name == "mock"


def test_sarvam_api_key_is_secret():
    assert "sarvam_api_key" in Settings.secret_fields()


def test_sarvam_fields_are_hidden_for_other_providers():
    meta = Settings.field_meta("sarvam_api_key")
    assert meta["depends_on"] == {"stt_provider": ["sarvam"], "tts_provider": ["sarvam"]}


@pytest.mark.parametrize(
    "key", ["sarvam_api_key", "sarvam_stt_model", "sarvam_tts_model", "sarvam_voice"]
)
def test_reload_watches_the_new_keys(key):
    source = inspect.getsource(Services.reload)
    assert key in source, f"{key} missing from a changed(...) list in Services.reload"


def test_importing_sarvam_modules_needs_no_sdk():
    """Provider modules must import on a bare install — the HTTP client is
    constructed in start(), not at module scope."""
    import vaani.providers.stt.sarvam as stt_mod
    import vaani.providers.tts.sarvam as tts_mod

    assert stt_mod.SarvamSTT is not None
    assert tts_mod.SarvamTTS is not None


def test_the_configured_voice_reaches_the_provider():
    s = Settings(tts_provider="sarvam", sarvam_api_key="k", sarvam_voice="bn-IN:anushka")
    assert build_tts(s)._default_voice == "bn-IN:anushka"
