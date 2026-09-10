"""Tata Smartflo voice-bot streaming: codec, transport and endpoints.

Everything here runs against mock providers and a fake Smartflo client, so the
wire format is exercised without a Tata account, a phone line or a network.
"""

from __future__ import annotations

from vaani.config import Settings


def test_smartflo_settings_carry_ui_metadata_and_hide_the_secret():
    fields = Settings.model_fields

    assert fields["smartflo_enabled"].default is False
    assert fields["smartflo_agent"].default == "default"

    for name in (
        "smartflo_enabled",
        "smartflo_webhook_secret",
        "smartflo_public_host",
        "smartflo_agent",
    ):
        extra = fields[name].json_schema_extra
        assert extra["group"] == "Telephony", f"{name} must render under Telephony"
        assert extra["label"], f"{name} needs a label for the admin page"
        # These take effect per call; none of them rebuilds a provider.
        assert extra["restart"] is False, f"{name} must not demand a restart"

    secret = fields["smartflo_webhook_secret"].json_schema_extra
    assert secret["secret"] is True, "the webhook secret must never be echoed back"
    assert fields["smartflo_public_host"].json_schema_extra["secret"] is False
