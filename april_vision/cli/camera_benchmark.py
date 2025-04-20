"""
Open cameras and measures the performance.

Measures the FPS for different resolutions.
"""
import argparse
import logging
from typing import List, NamedTuple

import cv2
from tabulate import tabulate

from ..detect_cameras import find_cameras

LOGGER = logging.getLogger(__name__)

standardResolutions = [
    (160, 120),

    (192, 144),
    (256, 144),

    (240, 160),

    (320, 240),
    (360, 240),
    (384, 240),
    (400, 240),
    (432, 240),

    (480, 320),

    (480, 360),
    (640, 360),

    (600, 480),
    (640, 480),
    (720, 480),
    (768, 480),
    (800, 480),
    (854, 480),
    (960, 480),

    (675, 540),
    (960, 540),

    (720, 576),
    (768, 576),
    (1024, 576),

    (750, 600),
    (800, 600),
    (1024, 600),

    (960, 640),
    (1024, 640),
    (1136, 640),

    (960, 720),
    (1152, 720),
    (1280, 720),
    (1440, 720),

    (960, 768),
    (1024, 768),
    (1152, 768),
    (1280, 768),
    (1366, 768),

    (1280, 800),

    (1152, 864),
    (1280, 864),
    (1536, 864),

    (1200, 900),
    (1440, 900),
    (1600, 900),

    (1280, 960),
    (1440, 960),
    (1536, 960),

    (1280, 1024),
    (1600, 1024),

    (1400, 1050),
    (1680, 1050),

    (1440, 1080),
    (1920, 1080),
    (2160, 1080),
    (2280, 1080),
    (2560, 1080),

    (2048, 1152),

    (1500, 1200),
    (1600, 1200),
    (1920, 1200),

    (1920, 1280),
    (2048, 1280),

    (1920, 1440),
    (2160, 1440),
    (2304, 1440),
    (2560, 1440),
    (2880, 1440),
    (2960, 1440),
    (3040, 1440),
    (3120, 1440),
    (3200, 1440),
    (3440, 1440),
    (5120, 1440),

    (2048, 1536),

    (2400, 1600),
    (2560, 1600),
    (3840, 1600),

    (2880, 1620),

    (2880, 1800),
    (3200, 1800),

    (2560, 1920),
    (2880, 1920),
    (3072, 1920),

    (2560, 2048),
    (2732, 2048),
    (3200, 2048),

    (2880, 2160),
    (3240, 2160),
    (3840, 2160),
    (4320, 2160),
    (5120, 2160),

    (3200, 2400),
    (3840, 2400),

    (3840, 2560),
    (4096, 2560),

    (5120, 2880),
    (5760, 2880),

    (4096, 3072),

    (7680, 4320),
    (10240, 4320),
]


class CameraBenchmarkResult(NamedTuple):
    """Camera benchmark result."""

    height: int
    width: int
    fps: float


def main(args: argparse.Namespace) -> None:
    """Measure the FPS for different resolutions."""
    if args.id is not None:
        camera_index = args.id
    elif args.serial is not None:
        cameras = find_cameras([], include_uncalibrated=True)
        cameras_serial_dict = {
            camera.serial_num: camera
            for camera in cameras if camera.serial_num
        }
        try:
            camera_index = cameras_serial_dict[args.serial].index
        except KeyError:
            raise ValueError(f"Camera with serial {args.serial} not found")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open camera with index {camera_index}")

    if args.set_codec is not None:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*args.set_codec))
        result = camera.get(cv2.CAP_PROP_FOURCC)
        if result != cv2.VideoWriter.fourcc(*args.set_codec):
            raise RuntimeError(f"Failed to set codec {args.set_codec} on camera")

    results: List[CameraBenchmarkResult] = []

    for resolution in standardResolutions:
        LOGGER.debug(f"Testing resolution: {resolution[0]}x{resolution[1]}")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        actual_resolution = (
            int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual_resolution != resolution:
            continue
        LOGGER.debug(f"Set resolution: {actual_resolution[0]}x{actual_resolution[1]}")

        # Measure FPS
        camera.set(cv2.CAP_PROP_FPS, 1000)
        fps = camera.get(cv2.CAP_PROP_FPS)

        results.append(CameraBenchmarkResult(
            height=actual_resolution[1],
            width=actual_resolution[0],
            fps=fps,
        ))
        LOGGER.debug(f"FPS: {fps}")

    camera.release()

    print(tabulate(results, headers="keys"))


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Camera_benchmark command parser."""
    parser = subparsers.add_parser(
        "camera_benchmark",
        description=(
            "Measure the maximum available FPS for the camera at different resolutions."),
        help="Measure the maximum available FPS for the camera at different resolutions."
    )

    cam_select_group = parser.add_mutually_exclusive_group(required=True)
    cam_select_group.add_argument(
        "--serial", type=str, default=None,
        help="Select the camera by serial number.")
    cam_select_group.add_argument(
        "--id", type=int, default=None, help="Select the camera index to use.")

    parser.add_argument(
        '--set_codec',
        type=str,
        default=None,
        help="4-character code of codec to set camera to (e.g. MJPG)"
    )

    parser.set_defaults(func=main)
