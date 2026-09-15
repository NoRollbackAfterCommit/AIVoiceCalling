"""Operator commands that need no browser.

The first administrator has to come from somewhere, and it must not be a
migration: a migration that creates an account with a known password is a
backdoor in every deployment that runs it. This creates one deliberately, and
only while there are no accounts at all.

    python -m vaani.admin create-admin admin@example.org
    python -m vaani.admin create-admin admin@example.org --password "..."
    python -m vaani.admin list-users

The password is read from a prompt rather than taken as an argument by default,
because an argument lands in the shell history and in `ps` for every other user
on the box.
"""

from __future__ import annotations

import asyncio
import getpass
import sys

from vaani.db.accounts import AccountRepository
from vaani.db.repository import CallRepository
from vaani.security.passwords import MIN_LENGTH
from vaani.settings_store import SettingsStore


async def _with_accounts(fn):
    settings = SettingsStore().settings
    repository = CallRepository(settings.database_url)
    await repository.start()
    try:
        return await fn(AccountRepository(repository.sessions))
    finally:
        await repository.close()


def _read_password() -> str:
    first = getpass.getpass("Password: ")
    if len(first) < MIN_LENGTH:
        print(f"The password must be at least {MIN_LENGTH} characters.", file=sys.stderr)
        raise SystemExit(2)
    if first != getpass.getpass("Repeat password: "):
        print("The passwords do not match.", file=sys.stderr)
        raise SystemExit(2)
    return first


async def _create_admin(email: str, password: str | None) -> int:
    secret = password or _read_password()

    async def run(accounts: AccountRepository) -> int:
        user = await accounts.ensure_first_admin(email, secret)
        if user is None:
            print(
                "There is already at least one account, so this command will not create "
                "another administrator. Sign in and add users from the portal, or ask an "
                "existing administrator to.",
                file=sys.stderr,
            )
            return 1
        print(f"Created platform administrator {user.email}.")
        return 0

    return await _with_accounts(run)


async def _list_users() -> int:
    async def run(accounts: AccountRepository) -> int:
        users = await accounts.users()
        if not users:
            print("No accounts yet. Create the first with: create-admin <email>")
            return 0
        print(f"{'email':<38} {'role':<16} {'organisation':>12}  status")
        for u in users:
            org = "—" if u.organisation_id is None else str(u.organisation_id)
            print(f"{u.email:<38} {u.role:<16} {org:>12}  {'active' if u.active else 'suspended'}")
        return 0

    return await _with_accounts(run)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    command, rest = args[0], args[1:]
    if command == "create-admin":
        if not rest:
            print("Usage: create-admin <email> [--password <password>]", file=sys.stderr)
            return 2
        email = rest[0]
        password = None
        if "--password" in rest:
            index = rest.index("--password")
            if index + 1 >= len(rest):
                print("--password needs a value", file=sys.stderr)
                return 2
            password = rest[index + 1]
        return asyncio.run(_create_admin(email, password))

    if command == "list-users":
        return asyncio.run(_list_users())

    print(f"Unknown command {command!r}. Try --help.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
