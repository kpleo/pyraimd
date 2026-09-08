"""Pyramid: Python wrapped Ab initio Molecular Dynamics."""

__version__ = "0.3.0"


def main() -> None:
    """Console entry point for the ``pyramid`` / ``pyraimd2`` commands.

    Imports are deferred so ``import pyraimd2`` stays free of heavy modules.
    """
    import sys

    from pyraimd2.cli import main as cli_main

    sys.exit(cli_main())
