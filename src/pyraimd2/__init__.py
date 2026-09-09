"""Pyramid: Python wrapped Ab initio Molecular Dynamics."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    # pyproject.toml is the single source of truth for the version.
    __version__ = _package_version("pyraimd2")
except PackageNotFoundError:  # running from a source tree without metadata
    __version__ = "0.4.1"


def main() -> None:
    """Console entry point for the ``pyramid`` / ``pyraimd2`` commands.

    Imports are deferred so ``import pyraimd2`` stays free of heavy modules.
    """
    import sys

    from pyraimd2.cli import main as cli_main

    sys.exit(cli_main())
