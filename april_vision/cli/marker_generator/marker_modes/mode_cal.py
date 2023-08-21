import argparse
import logging

import numpy as np
from PIL import Image

from april_vision.cli.utils import get_tag_family

from ..marker_tile import generate_tag_array, mm_to_pixels
from ..utils import DPI, PageSize

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    # Check the marker family has enough markers for the board
    required_markers = args.num_columns * args.num_rows
    if required_markers > tag_data.ncodes:
        LOGGER.error((
            f"Number of markers required ({required_markers}) is more than available ",
            f"in marker family ({tag_data.ncodes})"
        ))

    # Generate numpy arrays of all required markers
    marker_arrays = []
    for marker_id in range(required_markers):
        tag_array = generate_tag_array(tag_data, marker_id)
        marker_arrays.append(tag_array)

    # Calculate the dimentions of the full marker grid
    dim_width = ((tag_data.total_width - 1) * args.num_columns) + 1
    dim_height = ((tag_data.total_width - 1) * args.num_rows) + 1

    # Offset between each marker in the grid
    marker_offset = tag_data.total_width - 1

    # Create and fill np array of full marker board
    board_array = np.ones((dim_height, dim_width), dtype=np.uint8)

    for index, marker_array in enumerate(marker_arrays):
        row, col = divmod(index, args.num_columns)
        x = row * marker_offset
        y = col * marker_offset
        board_array[x:x + marker_array.shape[0], y:y + marker_array.shape[1]] = marker_array

    # Convert np array into correct size image
    marker_image = Image.fromarray(board_array)

    pixel_size = args.marker_size / tag_data.width_at_border

    total_board_pixel_width = (tag_data.total_width - 1) * args.num_columns + 1
    required_width = int(pixel_size * total_board_pixel_width)

    total_board_pixel_height = (tag_data.total_width - 1) * args.num_rows + 1
    required_height = int(pixel_size * total_board_pixel_height)

    resized_image = marker_image.resize(
        (mm_to_pixels(required_width), mm_to_pixels(required_height)),
        resample=0
    )

    # Paste board onto page and save
    page_size = PageSize[args.page_size]
    output_img = Image.new("RGB", page_size.pixels, (255, 255, 255))

    x_loc = (output_img.width - resized_image.width) // 2
    y_loc = (output_img.height - resized_image.height) // 2

    output_img.paste(resized_image, (x_loc, y_loc))

    filename = "cal_board_{}_{}_{}_{}.pdf".format(
        args.marker_family,
        args.num_rows,
        args.num_columns,
        args.marker_size,
    )

    output_img.save(
        filename,
        quality=100,
        dpi=(DPI, DPI),
    )
