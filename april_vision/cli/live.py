"""
Live camera demonstration.

Opens the camera and does live marker detection.
Can add overlays of fps, marker annotation, marker distance, etc..
Option to save the current view of the camera.
"""
import argparse
import logging
import os
from time import perf_counter

import cv2

from ..frame_sources import USBCamera
from ..marker import MarkerType
from ..vision import Processor
from ..utils import annotate_text, RollingAverage
from ..detect_cameras import find_cameras
from ..calibrations import calibrations

LOGGER = logging.getLogger(__name__)

# TODO add these functionality
# live
# 	distance


def main(args: argparse.Namespace):
    """Live camera demonstration."""
    avg_fps = RollingAverage(50)
    prev_frame_time = 0
    file_num = 1
    if args.id is None:
        cameras = find_cameras(calibrations, include_uncalibrated=True)
        try:
            camera = cameras[0]
        except IndexError:
            LOGGER.fatal("No cameras found")
            return
        source = USBCamera.from_calibration_file(
            camera.index, camera.calibration, camera.vidpid)
    else:
        source = USBCamera(args.id, (1280, 720))  # TODO open camera in a smarter way
    cam = Processor(
        source,
        tag_family=args.tag_family.value,
        quad_decimate=args.quad_decimate,
    )

    LOGGER.info("Press S to save image, press Q to exit")

    while True:
        frame = cam._capture()
        markers = cam._detect(frame)

        if args.annotate:
            cam._annotate(frame, markers, text_scale=0.5, line_thickness=2)

        new_frame_time = perf_counter()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        avg_fps.new_data(fps)
        if args.fps:
            frame = annotate_text(
                frame, f"{avg_fps.average():.0f}", (7, 70),
                text_scale=3, text_colour=(100, 255, 0))

        cv2.imshow('image', frame.colour_frame)

        button = cv2.waitKey(1) & 0xFF
        if button == ord('q'):
            break
        elif button == ord('s'):
            filename = f'saved_image{file_num:03d}.jpg'
            while os.path.exists(filename):
                file_num += 1
                filename = f'saved_image{file_num:03d}.jpg'

            cv2.imwrite(filename, frame.colour_frame)
            file_num += 1


def create_subparser(subparsers: argparse._SubParsersAction):
    """Live command parser."""
    parser = subparsers.add_parser(
        "live",
        description="Live camera demonstration with marker annotation.",
        help="Live camera demonstration with marker annotation.",
    )

    parser.add_argument(
        "--id", type=int, default=None, help="Override the camera index to use.")
    parser.add_argument(
        "--annotate", action='store_true', help="Annotate detected markers in the frames.")
    parser.add_argument(
        '--fps', action='store_true',
        help="Display the frames per second that the preview is running at.")

    parser.add_argument(
        '--tag_family', type=MarkerType, default=MarkerType.APRILTAG_36H11,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")
    parser.add_argument(
        '--quad_decimate', type=float, default=2,
        help="Set the level of decimation used in the detection stage")

    # parser.add_argument('--tag_size', type=int, default=0)
    # parser.add_argument('--distance', action='store_true')

    parser.set_defaults(func=main)
