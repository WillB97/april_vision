import argparse
import logging

import numpy as np
from PIL import Image

from april_vision.cli.utils import get_tag_family
from april_vision.marker import MarkerType

from ..marker_tile import generate_tag_array
from ..utils import parse_marker_ranges

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    marker_ids = parse_marker_ranges(tag_data, args.range)

    for marker_id in marker_ids:
        LOGGER.info(f"Creating marker: {marker_id}")
        tag_array = generate_tag_array(tag_data, marker_id)

        if args.aruco_orientation:
            # Rotate by 180deg to match the aruco format
            tag_array = np.rot90(tag_array, k=2)

        marker_image = Image.fromarray(tag_array)

        resized_image = marker_image.resize(
            (args.image_size, args.image_size),
            resample=0,
        )

        filename = args.filename.format(
            id=marker_id,
            marker_family=args.marker_family
        )
        resized_image.save(filename, quality=100)


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("IMAGE")

    parser.add_argument(
        "--marker_family", default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'",
    )
    parser.add_argument(
        "--range",
        help="Marker ids to output, can use '-' or ',' to specify lists and ranges",
        default="ALL",
    )
    parser.add_argument(
        "--image_size",
        help="The size of the output marker in pixels (default: %(default)s)",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--filename",
        type=str,
        help=(
            "Output filename of split files. `id` available for string format replacement "
            "(default: %(default)s)"
        ),
        default="{id}.png",
    )
    parser.add_argument(
        "--aruco_orientation",
        help="Rotate marker 180 for aruco orientation",
        action="store_true",
    )

    parser.set_defaults(func=main)
