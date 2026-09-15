"""User accounts: passwords, roles, and which organisation a person belongs to.

Until now anybody holding the shared bearer token was a full administrator of
every organisation on the deployment. With several customers' call centres on
one box that is not a policy, it is an accident waiting for an audit.
"""

from __future__ import annotations

import re

import pytest

from vaani.db.accounts import AccountRepository, Role
from vaani.db.repository import CallRepository
from vaani.db.tenancy import TenancyRepository
from vaani.security.passwords import hash_password, needs_rehash, verify_password

# -- password hashing ------------------------------------------------------


def test_a_password_is_never_stored_in_a_readable_form():
    stored = hash_password("correct horse battery staple")
    assert "correct horse" not in stored
    assert verify_password("correct horse battery staple", stored)


def test_the_wrong_password_is_refused():
    stored = hash_password("s3cret")
    assert not verify_password("s3cret ", stored)
    assert not verify_password("S3cret", stored)
    assert not verify_password("", stored)


def test_the_same_password_hashes_differently_every_time():
    """A shared salt would let one stolen hash identify every account using that
    password across the deployment."""
    assert hash_password("same") != hash_password("same")


def test_a_corrupt_or_foreign_hash_refuses_rather_than_raising():
    """A row hand-edited in the database, or written by an older scheme, must
    fail the login — not crash the login endpoint for everybody."""
    for stored in ["", "not-a-hash", "scrypt$bad$fields", "$2b$12$whatever"]:
        assert not verify_password("anything", stored)


def test_a_hash_from_the_current_scheme_does_not_need_rehashing():
    assert not needs_rehash(hash_password("x"))
    assert needs_rehash("some-older-scheme$abc")


# -- accounts --------------------------------------------------------------


@pytest.fixture
async def world(tmp_path):
    calls = CallRepository(f"sqlite+aiosqlite:///{tmp_path / 'accounts.db'}")
    await calls.start()
    tenancy = TenancyRepository(calls.sessions)
    accounts = AccountRepository(calls.sessions)
    org = await tenancy.create_organisation(slug="health", name="Health University")
    yield accounts, org
    await calls.close()


async def test_a_user_can_be_created_and_then_authenticates(world):
    accounts, org = world
    await accounts.create_user(
        email="admin@health.example",
        password="a-long-enough-password",
        name="Dr Admin",
        role=Role.ORG_ADMIN,
        organisation_id=org.id,
    )
    user = await accounts.authenticate("admin@health.example", "a-long-enough-password")
    assert user is not None
    assert user.role == Role.ORG_ADMIN
    assert user.organisation_id == org.id


async def test_the_wrong_password_does_not_authenticate(world):
    accounts, org = world
    await accounts.create_user(
        email="a@b.example",
        password="right-password-here",
        role=Role.VIEWER,
        organisation_id=org.id,
    )
    assert await accounts.authenticate("a@b.example", "wrong-password-here") is None


async def test_an_unknown_email_does_not_authenticate(world):
    accounts, _ = world
    assert await accounts.authenticate("nobody@nowhere.example", "whatever-password") is None


async def test_email_is_matched_without_regard_to_case_or_padding(world):
    """People type their address the way their mail client shows it."""
    accounts, org = world
    await accounts.create_user(
        email="Admin@Health.Example",
        password="a-long-enough-password",
        role=Role.ORG_ADMIN,
        organisation_id=org.id,
    )
    assert await accounts.authenticate("  admin@health.example ", "a-long-enough-password")


async def test_a_suspended_user_cannot_log_in(world):
    """Revoking access must not require deleting the person, or the calls and
    changes attributed to them lose their owner."""
    accounts, org = world
    user = await accounts.create_user(
        email="leaver@health.example",
        password="a-long-enough-password",
        role=Role.SUPERVISOR,
        organisation_id=org.id,
    )
    await accounts.set_active(user.id, False)
    assert await accounts.authenticate("leaver@health.example", "a-long-enough-password") is None


async def test_a_platform_admin_belongs_to_no_organisation(world):
    accounts, _ = world
    user = await accounts.create_user(
        email="root@euphoria.example",
        password="a-long-enough-password",
        role=Role.PLATFORM_ADMIN,
        organisation_id=None,
    )
    assert user.organisation_id is None


async def test_every_role_except_platform_admin_must_name_an_organisation(world):
    """A supervisor with no organisation would scope to nothing — or, worse, to
    everything, depending on which way the filter was written."""
    accounts, _ = world
    with pytest.raises(ValueError, match="organisation"):
        await accounts.create_user(
            email="x@y.example",
            password="a-long-enough-password",
            role=Role.SUPERVISOR,
            organisation_id=None,
        )


async def test_a_duplicate_email_is_refused(world):
    accounts, org = world
    await accounts.create_user(
        email="dup@health.example",
        password="a-long-enough-password",
        role=Role.VIEWER,
        organisation_id=org.id,
    )
    with pytest.raises(ValueError, match=re.escape("dup@health.example")):
        await accounts.create_user(
            email="DUP@health.example",
            password="another-password-here",
            role=Role.VIEWER,
            organisation_id=org.id,
        )


async def test_a_short_password_is_refused_at_creation(world):
    accounts, org = world
    with pytest.raises(ValueError, match="password"):
        await accounts.create_user(
            email="weak@health.example",
            password="short",
            role=Role.VIEWER,
            organisation_id=org.id,
        )


async def test_changing_a_password_invalidates_the_old_one(world):
    accounts, org = world
    user = await accounts.create_user(
        email="rotate@health.example",
        password="the-first-password",
        role=Role.VIEWER,
        organisation_id=org.id,
    )
    await accounts.set_password(user.id, "the-second-password")
    assert await accounts.authenticate("rotate@health.example", "the-first-password") is None
    assert await accounts.authenticate("rotate@health.example", "the-second-password")


async def test_the_first_admin_is_only_created_when_there_are_no_users(world):
    """Bootstrapping must not be a way to mint an administrator on a live box."""
    accounts, _ = world
    created = await accounts.ensure_first_admin("root@euphoria.example", "a-long-enough-password")
    assert created is not None

    again = await accounts.ensure_first_admin("second@euphoria.example", "a-long-enough-password")
    assert again is None
    assert await accounts.authenticate("second@euphoria.example", "a-long-enough-password") is None
