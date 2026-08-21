"""Trivial import smoke tests (M0). Real tests arrive with the loop modules."""

import pyraimd2


def test_import_and_version() -> None:
    assert pyraimd2.__version__ == "0.2.0"


def test_cli_entrypoint_exists() -> None:
    assert callable(pyraimd2.main)
