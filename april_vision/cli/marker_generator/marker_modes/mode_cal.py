"""Marker_generator subparser CAL_BOARD used to generate a calibration board."""
import argparse
import logging

import reportlab.graphics.shapes as rl_shapes
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors as rl_colors

from april_vision.cli.utils import get_tag_family
from april_vision.marker import MarkerType

from ..marker_tile import MarkerTileVector, mm_to_pixels
from ..utils import DEFAULT_VEC_FONT, DEFAULT_VEC_FONT_SIZE, DPI, PageSize

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate a calibration board."""
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    # Check the marker family has enough markers for the board
    required_markers = args.num_columns * args.num_rows
    if required_markers > tag_data.ncodes:
        LOGGER.error((
            f"Number of markers required ({required_markers}) is more than available ",
            f"in marker family ({tag_data.ncodes})"
        ))
        exit(1)

    # Generate numpy arrays of all required markers
    LOGGER.info("Generating markers for calibration board")

    marker_tiles: list[MarkerTileVector] = []
    for marker_id in range(required_markers):
        image_tile = MarkerTileVector(
            tag_data,
            marker_id,
            args.marker_size,
        )
        marker_tiles.append(image_tile)

    # Offset between each marker in the grid
    pixel_size = mm_to_pixels(args.marker_size / tag_data.width_at_border)
    marker_offset = (tag_data.total_width - 1) * pixel_size

    # Create a blank board
    page_size = PageSize[args.page_size]
    output_img = Drawing(page_size.vec_pixels.x, page_size.vec_pixels.y)

    # Calculate border size to center the markers on the page
    x_border = (page_size.pixels.x - (marker_offset * (args.num_columns - 1))) / 2
    y_border = (page_size.pixels.y - (marker_offset * (args.num_rows - 1))) / 2

    for index, marker_tile in enumerate(marker_tiles):
        row, col = divmod(index, args.num_columns)

        x = col * marker_offset
        y = row * marker_offset
        x_loc = x + x_border
        y_loc = y + y_border
        marker_tile.set_marker_centre(x_loc, y_loc)
        output_img.add(marker_tile.vectors)

    # Overlay info about the board
    text_overlay = "Family: {}  Rows: {}  Columns: {}  Marker size: {}".format(
        args.marker_family,
        args.num_rows,
        args.num_columns,
        args.marker_size,
    )
    text_offset = tag_data.total_width * pixel_size / 2
    output_img.add(rl_shapes.String(
        x_border - text_offset, y_border - text_offset - pixel_size * 0.6,
        text_overlay,
        fontSize=args.description_size,
        fontName=DEFAULT_VEC_FONT,
        textAnchor='start',
        fillColor=rl_colors.black,
    ))

    # Save file
    filename = "cal_board_{}_{}_{}_{}_{}.pdf".format(
        args.page_size,
        args.marker_family,
        args.num_rows,
        args.num_columns,
        args.marker_size,
    )

    # canvas DPI is 72
    output_img.scale(72 / DPI, 72 / DPI)
    renderPDF.drawToFile(output_img, filename)

    LOGGER.info(f"Calibration board saved as '{filename}'")


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_generator subparser CAL_BOARD used to generate a calibration board."""
    parser = subparsers.add_parser("CAL_BOARD", help="Generate a calibration board")

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
        default=DEFAULT_VEC_FONT_SIZE,
        type=int,
    )

    parser.set_defaults(func=main)
