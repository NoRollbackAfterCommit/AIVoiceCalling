"""One canonical spelling for a phone number.

Which organisation answers a call is decided by looking the dialled number up in
a table, so both sides of that comparison have to agree on how a number is
written — and nothing upstream guarantees it. The live Smartflo handshake sent
`" 918065605873"`, with a leading space and no plus, while an operator types
`+91 80 6560 5873` into the portal. Both must find the same row, or a caller
reaches the wrong organisation's agent.

Deliberately not a general phone-number library. `phonenumbers` would validate
against real numbering plans, and is the right answer the day this platform
handles more than a handful of countries — but it is a large dependency for a
base install that must work air-gapped, and the rules below cover every form the
carrier and the portal actually produce.
"""

from __future__ import annotations

# The deployment is Indian. A ten-digit number typed without a country code is
# the one ambiguous case, and assuming the deployment's own country is the only
# reading that is ever right; the parameter exists so that assumption is stated
# at the call site rather than buried here.
DEFAULT_COUNTRY = "91"

# Punctuation an operator or a carrier might include. Anything outside this set
# and the digits is a refusal, not something to strip: quietly discarding an
# unexpected character can turn one valid number into a different valid number.
_PUNCTUATION = " -() .\t"

# Shorter than this is an extension, a short code, or a typo — never a DID.
_MIN_NATIONAL = 10


def to_e164(raw: str | None, *, default_country: str = DEFAULT_COUNTRY) -> str | None:
    """`+` followed by digits, or None when the input cannot be trusted.

    None rather than a best guess: a wrong guess routes a real caller to another
    organisation's agent, which is worse than the fallback path that an unknown
    number already takes.
    """
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    plus = text.startswith("+")
    if plus:
        text = text[1:]

    for ch in _PUNCTUATION:
        text = text.replace(ch, "")

    if not text.isdigit():
        return None

    # 00 is the international access prefix, meaning the same thing as `+`. A
    # number that already said `+` is not also using it.
    if not plus and text.startswith("00"):
        text = text[2:]
        plus = True

    # A single leading 0 is the trunk prefix, how a national number is printed on
    # letterheads and in carrier portals. It is not part of the number, and left
    # in place it produces a country code of 0. Only stripped when what remains
    # is exactly a national number, so a genuine country code beginning with 0
    # — there are none today, but the check costs nothing — is never eaten.
    if not plus and text.startswith("0") and len(text) == _MIN_NATIONAL + 1:
        text = text[1:]

    if len(text) < _MIN_NATIONAL:
        return None

    # Exactly national length and no country code given: it belongs to the
    # deployment's own country. Longer than that already carries one.
    if not plus and len(text) == _MIN_NATIONAL:
        text = f"{default_country}{text}"

    return f"+{text}"
