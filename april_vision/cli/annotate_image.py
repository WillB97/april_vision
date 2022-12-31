"""
Annotate an image file.

Takes an input image, adds annotation and saves the image.
"""
import argparse
import logging
import os
from pathlib import Path

import cv2

from ..frame_sources import ImageSource
from ..marker import MarkerType
from ..vision import Processor

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    """Annotate an image file using the provided args."""
    input_file = Path(args.input_file)
    if not input_file.exists():
        LOGGER.fatal("Input file does not exist.")
        return

    source = ImageSource(args.input_file)
    threads = os.cpu_count()
    if threads is None:
        threads = 4
    processer = Processor(
        source, threads=threads,
        tag_family=args.tag_family.value, quad_decimate=args.quad_decimate,
    )

    frame = processer._capture()
    markers = processer._detect(frame)
    LOGGER.info(f"Found {len(markers)} markers.")
    frame = processer._annotate(frame, markers)

    # save frame
    cv2.imwrite(args.output_file, frame.colour_frame)

    if args.preview:
        cv2.imshow('image', frame.colour_frame)
        LOGGER.info("Press any key to close window.")
        _ = cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_subparser(subparsers: argparse._SubParsersAction):
    """Annotate_image command parser."""
    parser = subparsers.add_parser(
        "annotate_image",
        description="Annotate an image file with its markers",
        help="Annotate an image file with its markers",
    )

    parser.add_argument("input_file", type=str, help="The image to process.")
    parser.add_argument("output_file", type=str, help="The filepath to save the output to.")

    parser.add_argument('--preview', action='store_true', help="Display the annotated image.")

    parser.add_argument(
        '--tag_family', type=MarkerType, default=MarkerType.APRILTAG_36H11,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")
    parser.add_argument(
        '--quad_decimate', type=float, default=2,
        help="Set the level of decimation used in the detection stage")

    # parser.add_argument('--calibration', type=Path, default=None)
    # parser.add_argument('--tag_sizes', default=None)

    parser.set_defaults(func=main)
