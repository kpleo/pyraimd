"""Package import and version smoke tests."""

import pyraimd2


def test_import_and_version() -> None:
    assert pyraimd2.__version__ == "0.3.0"


def test_cli_entrypoint_exists() -> None:
    assert callable(pyraimd2.main)
