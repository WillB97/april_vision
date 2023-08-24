import argparse
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from april_vision.cli.utils import get_tag_family
from april_vision.marker import MarkerType

from ..marker_tile import generate_tag_array, mm_to_pixels
from ..utils import DEFAULT_FONT, DEFAULT_FONT_SIZE, DPI, PageSize

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate a calibration board"""
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
    LOGGER.info("Generating markers for calibration board")

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
        row = args.num_rows - (row + 1)

        x = row * marker_offset
        y = col * marker_offset
        board_array[x:x + marker_array.shape[0], y:y + marker_array.shape[1]] = marker_array

    # Convert np array into correct size image
    marker_image = Image.fromarray(board_array)

    pixel_size = args.marker_size / tag_data.width_at_border

    required_width = int(pixel_size * dim_width)
    required_height = int(pixel_size * dim_height)

    resized_image = marker_image.resize(
        (mm_to_pixels(required_width), mm_to_pixels(required_height)),
        resample=0
    )

    # Paste board onto page
    page_size = PageSize[args.page_size]
    output_img = Image.new("RGB", page_size.pixels, (255, 255, 255))

    x_loc = (output_img.width - resized_image.width) // 2
    y_loc = (output_img.height - resized_image.height) // 2

    output_img.paste(resized_image, (x_loc, y_loc))

    # Overlay info about the board
    text_overlay = "Family: {}  Rows: {}  Columns: {}  Marker size: {}".format(
        args.marker_family,
        args.num_rows,
        args.num_columns,
        args.marker_size,
    )

    image_draw = ImageDraw.Draw(output_img)
    image_draw.text(
        (x_loc, y_loc + resized_image.height),
        text_overlay,
        fill="black",
        anchor="lt",
        font=ImageFont.truetype(DEFAULT_FONT, args.description_size),
    )

    # Save file
    filename = "cal_board_{}_{}_{}_{}_{}.pdf".format(
        args.page_size,
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

    LOGGER.info(f"Calibration board saved as '{filename}'")


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_generator subparser CAL_BOARD used to generate a calibration board."""
    parser = subparsers.add_parser("CAL_BOARD")

    parser.add_argument(
        "--page_size",
        type=str,
        help="Page size. (default: %(default)s)",
        choices=sorted([size.name for size in PageSize]),
        default="A4L",
    )

    parser.add_argument(
        '--marker_family',
        default=MarkerType.APRILTAG_36H11.value,
        choices=[
            MarkerType.APRILTAG_16H5.value,
            MarkerType.APRILTAG_25H9.value,
            MarkerType.APRILTAG_36H11.value,
        ],
        help="Set the marker family used in the calibration board (default: %(default)s)",
    )

    parser.add_argument(
        "--marker_size",
        help="The size of markers in millimeters (default: %(default)s)",
        default=40,
        type=int,
    )
    parser.add_argument(
        "--num_columns",
        help="Number of columns on calibration board (default: %(default)s)",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--num_rows",
        help="Number of rows on calibration board (default: %(default)s)",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--description_size",
        help="Set the text size (default: %(default)s)",
        default=DEFAULT_FONT_SIZE,
        type=int,
    )

    parser.set_defaults(func=main)
