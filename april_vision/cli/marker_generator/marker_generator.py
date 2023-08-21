"""
Generate marker PDFs.

Takes a set of parameters and generates marker PDFs.
"""
import argparse
import importlib
import logging

LOGGER = logging.getLogger(__name__)

DPI = 72
BORDER_FILL = "lightgrey"


generation_modes = [
    "mode_single",
    "mode_tile",
    "mode_image",
    "mode_cal",
]


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_generator command parser."""

    parser = subparsers.add_parser(
        "marker_generator",
        description="Generate markers",
        help="Generate markers",
    )

    marker_subparsers = parser.add_subparsers(required=True, help="Marker generation mode")

    for mode in generation_modes:
        mod_name = f"{__package__}.marker_modes.{mode}"
        importlib.import_module(mod_name).create_subparser(marker_subparsers)
