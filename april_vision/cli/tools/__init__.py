import argparse
import importlib

subcommands = [
    "family_details",
    "list_cameras",
    "calibration_stats",
]


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Subparser for the tools commands."""
    parser = subparsers.add_parser(
        "tools",
        description="A collection of useful tools",
        help="A collection of useful tools",
    )

    subparsers = parser.add_subparsers(required=True)
    for command in subcommands:
        mod_name = f"{__package__}.{command}"
        importlib.import_module(mod_name).create_subparser(subparsers)
