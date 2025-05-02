"""Print statistics about a calibration file."""
import argparse
from math import atan, degrees
from pathlib import Path

from april_vision.utils import load_calibration_extra


def main(args: argparse.Namespace) -> None:
    """Print statistics about a calibration file."""
    # Load the calibration file
    try:
        calibration_data = load_calibration_extra(args.calibration)
    except FileNotFoundError:
        print(f"Calibration file {args.calibration} does not exist.")
        exit(1)
    except SystemError:
        print(f"Calibration file {args.calibration} is not a valid XML file.")
        exit(1)

    fx, fy, cx, cy = calibration_data['calibration']
    resolution = calibration_data['resolution']
    pidvid_list = calibration_data['vidpids']

    # Process values
    image_center = (resolution[0] / 2, resolution[1] / 2)
    optical_center_offset = (
        round(float(cx - image_center[0]), 2),
        round(float(cy - image_center[1]), 2),
    )

    # f = (resolution[0] / 2) / tan(fov / 2)
    hfov = degrees(2 * atan(resolution[0] / (2 * fx)))
    vfov = degrees(2 * atan(resolution[1] / (2 * fy)))

    if pidvid_list:
        pidvids = ', '.join(pidvid_list)
    else:
        pidvids = "Unknown"

    # Print the results
    print(f"Calibration file: {args.calibration}")
    print(f"Camera ID: {pidvids}")
    print(f"Resolution: {resolution}")
    print(f"Optical center offset: {optical_center_offset}")
    print(f"FOV, horizontal: {hfov:.2f} degrees, vertical: {vfov:.2f} degrees")
    print(f"Pixel squareness: {fx / fy:.4f} (1.0 is square, < 1.0 is wide, > 1.0 is tall)")


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Print statistics about a calibration file."""
    parser = subparsers.add_parser(
        "calibration_stats",
        description="Print statistics about a calibration file.",
        help="Print statistics about a calibration file.",
    )

    parser.add_argument(
        'calibration',
        help="The calibration file to analyze",
        type=Path,
    )

    parser.set_defaults(func=main)
