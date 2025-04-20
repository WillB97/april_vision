"""List out the available cameras connected to the system."""
import argparse

from tabulate import tabulate

from april_vision.calibrations import calibrations, extra_calibrations
from april_vision.detect_cameras import find_cameras


def main(args: argparse.Namespace) -> None:
    """List out the available cameras connected to the system."""
    if args.extra_calibrations:
        cameras = find_cameras(extra_calibrations, include_uncalibrated=True)
    else:
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

    parser.add_argument(
        '-e', '--extra-calibrations',
        help="Include the additional optional calibrations",
        action='store_true',
    )

    parser.set_defaults(func=main)
