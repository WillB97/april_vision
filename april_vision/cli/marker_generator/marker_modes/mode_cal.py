import argparse
import logging

from PIL import Image

from april_vision.cli.utils import get_tag_family

from ..marker_tile import MarkerTile
from ..utils import DPI, PageSize

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    required_markers = args.num_columns * args.num_rows
    if required_markers > tag_data.ncodes:
        LOGGER.error((
            f"Number of markers required ({required_markers}) is more than available ",
            f"in marker family ({tag_data.ncodes})"
        ))

    marker_tiles = []

    for marker_id in range(required_markers):
        image_tile = MarkerTile(
            tag_data,
            marker_id,
            args.marker_size,
            aruco_orientation=args.aruco_orientation
        )

        marker_tiles.append(image_tile)

    page_size = PageSize[args.page_size]
    output_img = Image.new("RGB", page_size.pixels, (255, 255, 255))

    for index, marker in enumerate(marker_tiles):
        row, col = divmod(index, args.num_columns)

        top_left_x = (output_img.width - (args.num_columns * marker.marker_width)) // 2
        x_loc = top_left_x + (col * marker.marker_width)

        top_left_y = (output_img.height - (args.num_rows * marker.marker_height)) // 2
        y_loc = top_left_y + (row * marker.marker_height)

        output_img.paste(marker.image, (x_loc, y_loc))

    # Save combined PDF
    combined_filename = args.all_filename.format(
        marker_family=args.marker_family
    )

    output_img.save(
        combined_filename,
        quality=100,
        dpi=(DPI, DPI),
    )
