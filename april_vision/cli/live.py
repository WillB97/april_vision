"""
Live camera demonstration.

Opens the camera and does live marker detection.
Can add overlays of fps, marker annotation, marker distance, etc..
Option to save the current view of the camera.
"""
import argparse
import os

import cv2

from ..frame_sources import USBCamera
from ..marker import MarkerType
from ..vision import Processor

# TODO add these functionality
# live
# 	anotate
# 	fps
# 	distance
#   save button


def main(args: argparse.Namespace):
    """Live camera demonstration."""
    source = USBCamera(args.id, (1280, 720))  # TODO open camera in a smarter way
    threads = os.cpu_count()
    if threads is None:
        threads = 4
    cam = Processor(
        source, threads=threads,
        tag_family=args.tag_family.value, quad_decimate=args.quad_decimate,
    )

    while True:
        frame = cam._capture()
        markers = cam._detect(frame)

        if args.annotate:
            cam._annotate(frame, markers, text_scale=0.5, line_thickness=2)

        cv2.imshow('image', frame.colour_frame)

        button = cv2.waitKey(1) & 0xFF
        if button == ord('q'):
            break


def create_subparser(subparsers: argparse._SubParsersAction):
    """Live command parser."""
    parser = subparsers.add_parser("live")

    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--annotate", action='store_true')
    # parser.add_argument('--fps', action='store_true')

    parser.add_argument(
        '--tag_family', type=MarkerType, default=MarkerType.APRILTAG_36H11,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")
    parser.add_argument(
        '--quad_decimate', type=float, default=2,
        help="Set the level of decimation used in the detection stage")

    # parser.add_argument('--calibration', type=Path, default=None)
    # parser.add_argument('--tag_sizes', default=None)
    # parser.add_argument('--distance', action='store_true')

    parser.set_defaults(func=main)
