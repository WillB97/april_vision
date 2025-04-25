"""Print statistics about a calibration file."""
import argparse
from math import atan, degrees
from pathlib import Path

import cv2


def main(args: argparse.Namespace) -> None:
    """Print statistics about a calibration file."""
    calibration_file: Path = args.calibration
    if not calibration_file.exists():
        print(f"Calibration file {args.calibration} does not exist.")
        exit(1)

    if not calibration_file.is_file():
        print(f"Calibration file {args.calibration} is not a file.")
        exit(1)

    # Load the calibration file
    storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)
    resolution_node = storage.getNode("cameraResolution")
    camera_matrix = storage.getNode("cameraMatrix").mat()
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Process values
    resolution = (
        int(resolution_node.at(0).real()),
        int(resolution_node.at(1).real()),
    )
    image_center = (resolution[0] / 2, resolution[1] / 2)
    optical_center_offset = (
        round(float(cx - image_center[0]), 2),
        round(float(cy - image_center[1]), 2),
    )

    # f = (resolution[0] / 2) / tan(fov / 2)
    hfov = degrees(2 * atan(resolution[0] / (2 * fx)))
    vfov = degrees(2 * atan(resolution[1] / (2 * fy)))

    vidpid_node = storage.getNode('vidpid')
    if vidpid_node.isSeq():
        pidvids = ', '.join([vidpid_node.at(i).string() for i in range(vidpid_node.size())])
    elif vidpid_node.isString():
        pidvids = vidpid_node.string()
    else:
        pidvids = "Unknown"

    storage.release()

    # Print the results
    print(f"Calibration file: {calibration_file}")
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
