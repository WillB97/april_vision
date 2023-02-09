import argparse
import importlib

subcommands = [
    "family_details",
    "list_cameras",
]


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "tools",
        description="A collection of useful tools",
        help="A collection of useful tools",
    )

    subparsers = parser.add_subparsers(required=True)
    for command in subcommands:
        mod_name = f"{__package__}.{command}"
        importlib.import_module(mod_name).create_subparser(subparsers)
