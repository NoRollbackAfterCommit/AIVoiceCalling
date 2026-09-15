"""Normalising a dialled number to one canonical form.

Every call's organisation is decided by looking the dialled number up in a
table, so the two sides of that comparison must agree on how a number is
written. The carrier does not help: the live Smartflo handshake logged the
number as `" 918065605873"` — no plus, and a leading space — while operators
type `+91 80 6560 5873` into the portal. Both must find the same row.
"""

from __future__ import annotations

import pytest

from vaani.telephony.numbers import to_e164


@pytest.mark.parametrize(
    "raw",
    [
        "+918065605873",
        "918065605873",
        " 918065605873",  # exactly what Tata's handshake sent on 15 Sep
        "918065605873 ",
        "+91 80 6560 5873",
        "+91-80-6560-5873",
        "+91 (80) 6560 5873",
        "00918065605873",  # international prefix instead of +
        "8065605873",  # ten digits, national form
        "08065605873",  # national form with the trunk prefix, as printed locally
        "0 80 6560 5873",
    ],
)
def test_every_way_this_number_is_written_reaches_one_form(raw):
    assert to_e164(raw) == "+918065605873"


def test_a_number_from_another_country_keeps_its_own_code():
    assert to_e164("+14155550123") == "+14155550123"
    assert to_e164("0014155550123") == "+14155550123"


def test_the_default_country_is_configurable_for_a_non_indian_deployment():
    assert to_e164("4155550123", default_country="1") == "+14155550123"


@pytest.mark.parametrize("raw", ["", "   ", None, "not a number", "+", "12", "extension 4"])
def test_nothing_usable_is_rejected_rather_than_guessed(raw):
    """A wrong guess routes a real caller to another organisation's agent."""
    assert to_e164(raw) is None


def test_letters_mixed_into_digits_are_refused_not_stripped():
    """Silently dropping characters could turn one valid number into another."""
    assert to_e164("+9180656O5873") is None  # letter O, not zero
