"""
Live camera demonstration.

Opens the camera and does live marker detection.
Can add overlays of fps, marker annotation, marker distance, etc..
Option to save the current view of the camera.
"""
import argparse
import logging
import os
from math import degrees
from time import perf_counter

import cv2

from ..calibrations import calibrations
from ..detect_cameras import find_cameras
from ..frame_sources import USBCamera
from ..marker import MarkerType
from ..utils import RollingAverage, annotate_text, normalise_marker_text
from ..vision import Processor

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Live camera demonstration."""
    avg_fps = RollingAverage(50)
    prev_frame_time: float = 0
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
        source = USBCamera(args.id, (1280, 720))
    cam = Processor(
        source,
        tag_family=args.tag_family,
        quad_decimate=args.quad_decimate,
        tag_sizes=float(args.tag_size) / 1000,
        calibration=source.calibration,
    )

    LOGGER.info("Press S to save image, press Q to exit")

    while True:
        frame = cam._capture()
        markers = cam._detect(frame)

        if args.annotate:
            cam._annotate(frame, markers)

        new_frame_time = perf_counter()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        avg_fps.new_data(fps)
        if args.fps:
            frame = annotate_text(
                frame, f"{avg_fps.average():.0f}", (7, 70),
                text_scale=3, text_colour=(100, 255, 0))

        if args.distance:
            for marker in markers:
                if marker.has_pose():
                    text_scale = normalise_marker_text(marker)

                    loc = (
                        int(marker.pixel_centre[0] - 80 * text_scale),
                        int(marker.pixel_centre[1] + 40 * text_scale),
                    )
                    frame = annotate_text(
                        frame, f"dist={marker.distance}mm", loc,
                        text_scale=0.8 * text_scale,
                        text_colour=(255, 191, 0),  # deep sky blue
                    )

        for marker in markers:
            output = "#{}, {}, ({}, {})".format(
                marker.id,
                marker.marker_type.value.strip('tag'),
                int(marker.pixel_centre.x),
                int(marker.pixel_centre.y),
            )

            if marker.has_pose():
                output += (
                    ", | b={} | r={}, θ={}, φ={} | "
                    "x={}, y={}, z={} | r={}, p={}, y={}"
                ).format(
                    int(marker.bearing),
                    int(marker.spherical.r),
                    int(degrees(marker.spherical.theta)),
                    int(degrees(marker.spherical.phi)),
                    int(marker.cartesian.x),
                    int(marker.cartesian.y),
                    int(marker.cartesian.z),
                    int(degrees(marker.orientation.roll)),
                    int(degrees(marker.orientation.pitch)),
                    int(degrees(marker.orientation.yaw)),
                )
            else:
                output += ", no values for pose estimation"

            LOGGER.info(output)

        cv2.imshow('image', frame.colour_frame)

        button = cv2.waitKey(1) & 0xFF
        if (button == ord('q')) or (button == 27):
            # Quit on q or ESC key
            break
        elif button == ord('s'):
            filename = f'saved_image{file_num:03d}.jpg'
            while os.path.exists(filename):
                file_num += 1
                filename = f'saved_image{file_num:03d}.jpg'

            cv2.imwrite(filename, frame.colour_frame)
            file_num += 1


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Live command parser."""
    parser = subparsers.add_parser(
        "live",
        description="Live camera demonstration with marker annotation.",
        help="Live camera demonstration with marker annotation.",
    )

    parser.add_argument(
        "--id", type=int, default=None, help="Override the camera index to use.")
    parser.add_argument(
        "--no_annotate", action='store_false', dest='annotate',
        help="Turn off marker annotation for detected markers.")
    parser.add_argument(
        '--fps', action='store_true',
        help="Display the frames per second that the preview is running at.")

    parser.add_argument(
        '--tag_family', default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")
    parser.add_argument(
        '--quad_decimate', type=float, default=2,
        help="Set the level of decimation used in the detection stage")

    parser.add_argument(
        '--tag_size', type=int, default=0, help="The size of markers in millimeters")
    parser.add_argument(
        '--distance', action='store_true',
        help="Annotate frames with the distance to the marker.")

    parser.set_defaults(func=main)
