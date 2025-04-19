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
from typing import List, Tuple

import cv2

from ..calibrations import calibrations
from ..detect_cameras import find_cameras
from ..frame_sources import USBCamera
from ..marker import MarkerType
from ..utils import RollingAverage, annotate_text, normalise_marker_text
from ..vision import Processor

LOGGER = logging.getLogger(__name__)


def parse_properties(args: argparse.Namespace) -> List[Tuple[int, int]]:
    """Parse the camera properties supplied on the command line."""
    props = []

    if args.set_fps is not None:
        props.append((cv2.CAP_PROP_FPS, args.set_fps))

    if args.set_codec is not None:
        props.append((cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*args.set_codec)))

    return props


def main(args: argparse.Namespace) -> None:
    """Live camera demonstration."""
    cap_time_avg = RollingAverage(20)
    proc_time_avg = RollingAverage(20)

    file_num = 1

    camera_properties = parse_properties(args)

    if args.id is None:
        cameras = find_cameras(calibrations, include_uncalibrated=True)
        try:
            camera = cameras[0]
        except IndexError:
            LOGGER.fatal("No cameras found")
            return
        source = USBCamera.from_calibration_file(
            camera.index,
            camera.calibration,
            camera.vidpid,
            camera_parameters=camera_properties,
        )
    else:
        if args.set_resolution is not None:
            resolution_raw = args.set_resolution.split('x')
            resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
        else:
            resolution = (1280, 720)
        source = USBCamera(
            args.id,
            resolution,
            camera_parameters=camera_properties,
        )
        print(f"Resolution set to {source._get_resolution()}")
    cam = Processor(
        source,
        tag_family=args.tag_family,
        quad_decimate=args.quad_decimate,
        tag_sizes=float(args.tag_size) / 1000,
        calibration=source.calibration,
        aruco_orientation=args.aruco_orientation,
    )

    LOGGER.info("Press S to save image, press Q to exit")

    while True:
        start_time = perf_counter()
        frame = cam._capture()
        cap_time = perf_counter()
        markers = cam._detect(frame)
        proc_time = perf_counter()

        cap_time_avg.new_data(1000 * (cap_time - start_time))
        proc_time_avg.new_data(1000 * (proc_time - cap_time))

        if args.annotate:
            cam._annotate(frame, markers)

        if args.perf:
            frame = annotate_text(
                frame,
                f"Capture: {cap_time_avg.average():.2f}ms",
                (10, 30),
                text_scale=0.75,
                text_colour=(100, 255, 0),
                thickness=1,
            )
            frame = annotate_text(
                frame,
                f"Detect: {proc_time_avg.average():.2f}ms",
                (10, 60),
                text_scale=0.75,
                text_colour=(100, 255, 0),
                thickness=1,
            )

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
        '--perf', action='store_true',
        help="Display the performance of the capture/detection in ms to do each operation.")

    parser.add_argument(
        '--tag_family', default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")
    parser.add_argument(
        '--aruco_orientation', action='store_true', help="Use ArUco marker orientation.")
    parser.add_argument(
        '--quad_decimate', type=float, default=2,
        help="Set the level of decimation used in the detection stage")

    parser.add_argument(
        '--tag_size', type=int, default=0, help="The size of markers in millimeters")
    parser.add_argument(
        '--distance', action='store_true',
        help="Annotate frames with the distance to the marker.")

    parser.add_argument(
        '--set_fps',
        type=int,
        default=None,
        help="The FPS to set the camera to"
    )
    parser.add_argument(
        '--set_codec',
        type=str,
        default=None,
        help="4-character code of codec to set camera to (e.g. MJPG)"
    )
    parser.add_argument(
        '--set_resolution',
        type=str,
        default=None,
        help="The resolution to use for a manually specified camera (e.g. 1280x720)"
    )

    parser.set_defaults(func=main)
