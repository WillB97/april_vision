import argparse

from tabulate import tabulate

from april_vision.calibrations import calibrations
from april_vision.detect_cameras import find_cameras


def main(args: argparse.Namespace) -> None:
    cameras = find_cameras(calibrations, include_uncalibrated=True)
    print(tabulate(cameras, headers="keys"))


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """List out the available cameras connected to the system."""
    parser = subparsers.add_parser(
        "list_cameras",
        description=(
            "List out the available cameras connected to the system, "
            "this will also search your current directory for calibration files"
        ),
        help=(
            "List out the available cameras connected to the system, "
            "this will also search your current directory for calibration files"
        ),
    )

    parser.set_defaults(func=main)
