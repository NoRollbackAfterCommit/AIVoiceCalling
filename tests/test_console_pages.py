"""The console's pages are served, and stay reachable without a token.

The shell of a page carries no data — every figure on it arrives over a guarded
`/api` call or the guarded `/ws/monitor` socket. Demanding the token for the
HTML too would leave an operator staring at a 401 with nowhere to type it, which
is why `auth.py` lets the pages through and guards what they fetch.
"""

from __future__ import annotations

import httpx

from vaani.main import create_app
from vaani.pipeline.manager import CallManager
from vaani.telephony.announce import CallAnnouncements

TOKEN = "s3cret-token"

# Every console page: the path it is served at, and a string from its own markup
# that proves the right file came back rather than index.html for all of them.
PAGES = [
    ("/", 'id="dtmf"'),
    ("/calls", 'id="roster"'),
    ("/login", 'autocomplete="current-password"'),
    ("/organisations", 'id="didForm"'),
    ("/users", 'id="roleHelp"'),
    ("/reports", 'id="figures"'),
    ("/knowledge", 'id="drop"'),
    ("/settings", 'id="dirty"'),
    ("/agents", 'id="editor"'),
]


def _client(services, settings):
    app = create_app(settings.model_copy(update={"api_token": TOKEN, "env": "dev"}))
    app.state.services = services
    app.state.calls = CallManager(max_concurrent=2)
    app.state.announcements = CallAnnouncements()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_every_console_page_is_served_without_a_token(services, settings):
    async with _client(services, settings) as c:
        for path, marker in PAGES:
            response = await c.get(path)
            assert response.status_code == 200, f"{path} returned {response.status_code}"
            assert marker.lower() in response.text.lower(), f"{path} served the wrong file"


async def test_the_live_call_page_reads_the_monitor_socket(services, settings):
    """The page is only a shell; if it does not open /ws/monitor it shows nothing.

    Asserting on the markup rather than on a browser because the repo has no JS
    harness — but the socket path and the roster container are the two things
    whose absence would make the page silently blank, so they are worth pinning.
    """
    async with _client(services, settings) as c:
        page = (await c.get("/calls")).text

    assert "/ws/monitor" in page, "the page must subscribe to the supervisor feed"
    assert "vaaniAuth.wsUrl" in page, "the socket must carry the token the guard requires"
    assert "/api/calls/" in page, "hang-up and stored detail both go through /api/calls"


# -- the console signs in; it does not hold the master key -------------------


def _auth_js() -> str:
    from pathlib import Path

    import vaani

    return (Path(vaani.__file__).parent / "web" / "static" / "auth.js").read_text(encoding="utf-8")


def test_the_console_never_attaches_the_shared_api_token():
    """The token is a machine credential — the deploy script, the health probe,
    the carrier harness. A browser holding one bypassed the sign-in screen
    entirely: writes succeeded while `/api/auth/me` said nobody was signed in,
    so pages rendered blank lists over a session that did not exist, and data
    could be fed in without anyone logging in.

    People carry a session cookie, which the browser attaches by itself.
    """
    source = _auth_js()
    assert "Authorization" not in source, "the console is still sending the shared token"
    assert "token=" not in source, "the console is still putting a token on the socket URL"


def test_the_console_discards_any_token_it_finds_stored():
    """A browser that was given one before this change must stop using it,
    without anybody having to clear their site data."""
    source = _auth_js()
    assert "removeItem" in source


def test_every_console_page_loads_the_auth_helper():
    """A page that forgets it has no sign-out control, no role-aware menu, and
    no way to notice the session has gone."""
    from pathlib import Path

    import vaani

    static = Path(vaani.__file__).parent / "web" / "static"
    for page in static.glob("*.html"):
        if page.name == "login.html":
            continue  # the one page that must work without a session
        assert "/static/auth.js" in page.read_text(encoding="utf-8"), page.name


# -- the agent editor --------------------------------------------------------


def _static(name: str) -> str:
    from pathlib import Path

    import vaani

    return (Path(vaani.__file__).parent / "web" / "static" / name).read_text(encoding="utf-8")


def test_the_agent_editor_reads_and_writes_the_agents_api():
    """Until this page, creating the agent that answers a line was the one setup
    step an operator could not do: it needed the API token, which means it
    needed an engineer. Four calls carry the page — two to show anything, one to
    save, and one so the form can only offer tools the server will accept."""
    page = _static("agents.html")
    assert '"/api/agents"' in page, "without the list the page has nothing to show"
    assert '"/api/agents/"' in page, "one agent must load into the form to be edited"
    assert '"PUT"' in page, "a form that cannot save is decoration"
    assert '"/api/tools"' in page, "PUT refuses an unknown tool name outright"


def test_a_new_agent_starts_from_the_servers_defaults_not_from_a_blank_form():
    """Caught in a browser before this was pinned: a new agent saved straight
    from an empty form stored `tools: []`, because a posted blank overrides the
    default it was meant to inherit. That agent could not search its own
    documents, could not transfer, and — end_call not running without
    set_disposition — could never hang up.
    """
    assert "/api/agent-defaults" in _static("agents.html")


def test_the_agent_key_cannot_be_edited_once_it_exists():
    """Knowledge is namespaced by agent key, so renaming one leaves its whole
    corpus addressed to a key nothing answers to. The agent would silently
    forget everything it had been taught, with no error raised anywhere."""
    assert "readOnly" in _static("agents.html")


def test_the_console_links_to_the_agent_editor_for_those_who_may_use_it():
    """A page nothing links to is a page nobody finds. The roles on the link
    must match the server's own rule for writes to /api/agents, or the console
    offers somebody a page that refuses them."""
    links = [ln for ln in _static("index.html").splitlines() if 'href="/agents"' in ln]
    assert links, "the console must link to the agent editor"
    assert 'data-roles="platform_admin,org_admin"' in links[0]
