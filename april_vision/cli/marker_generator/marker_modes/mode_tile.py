"""Marker generator mode to generate a PDF with multiple markers per page."""
import argparse
import logging

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.pdfgen import canvas

from april_vision.cli.utils import get_tag_family
from april_vision.marker import MarkerType

from ..marker_tile import MarkerTileVector
from ..utils import (
    DEFAULT_COLOUR,
    DEFAULT_VEC_FONT,
    DEFAULT_VEC_FONT_SIZE,
    DPI,
    PageSize,
    mm_to_pixels,
    parse_marker_ranges,
)

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate a page of multiple markers with the provided arguments."""
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    marker_ids = parse_marker_ranges(tag_data, args.range)

    marker_tiles: list[MarkerTileVector] = []

    for marker_id in marker_ids:
        image_tile = MarkerTileVector(
            tag_data,
            marker_id,
            args.marker_size,
            aruco_orientation=args.aruco_orientation
        )
        image_tile.add_border_line(
            args.border_width,
            args.border_colour,
        )
        image_tile.add_centre_ticks(
            args.border_width,
            args.tick_length,
            args.border_colour,
        )

        if args.no_number is False:
            image_tile.add_id_number(
                DEFAULT_VEC_FONT,
                args.number_size,
                args.border_colour,
            )

        image_tile.add_description_border(
            args.description_format,
            DEFAULT_VEC_FONT,
            args.description_size,
            "black",
        )

        for _ in range(args.repeat):
            marker_tiles.append(image_tile.copy())

    page_size = PageSize[args.page_size]

    # Converts list of markers into a list of lists of markers per page
    markers_per_page = args.num_columns * args.num_rows
    marker_tiles_for_page = [
        marker_tiles[n:n + markers_per_page]
        for n in range(0, len(marker_tiles), markers_per_page)
    ]

    combined_filename = args.all_filename.format(
        marker_family=args.marker_family
    )
    combined_pdf = canvas.Canvas(combined_filename, pagesize=page_size.vec_pixels)

    for markers in marker_tiles_for_page:
        output_img = Drawing(page_size.vec_pixels.x, page_size.vec_pixels.y)

        for index, marker in enumerate(markers):
            row, col = divmod(index, args.num_columns)

            column_spacing = mm_to_pixels(args.column_padding) + marker.marker_width
            row_spacing = mm_to_pixels(args.row_padding) + marker.marker_width
            outer_marker_centre_width = (
                (args.num_columns - 1) * column_spacing
            )
            outer_marker_centre_height = (
                (args.num_rows - 1) * row_spacing
            )

            if args.left_margin is not None:
                x_offset = mm_to_pixels(args.left_margin) + marker.marker_width / 2
            elif args.right_margin is not None:
                x_offset = (
                    page_size.pixels.x
                    - mm_to_pixels(args.right_margin)
                    - marker.marker_width / 2
                    - outer_marker_centre_width
                )
            else:
                # Centered
                x_offset = (
                    page_size.pixels.x
                    - outer_marker_centre_width
                ) / 2

            x_loc = x_offset + (col * column_spacing)

            if args.top_margin is not None:
                y_offset = (
                    page_size.pixels.y
                    - mm_to_pixels(args.top_margin)
                    - marker.marker_width / 2
                )
            elif args.bottom_margin is not None:
                y_offset = (
                    mm_to_pixels(args.bottom_margin)
                    + marker.marker_width / 2
                    + outer_marker_centre_height
                )
            else:
                # Centered
                y_offset = page_size.pixels.y - (
                    page_size.pixels.y
                    - outer_marker_centre_height
                ) / 2

            y_loc = y_offset - (row * row_spacing)

            marker.set_marker_centre(x_loc, y_loc)
            output_img.add(marker.vectors)

        # canvas DPI is 72
        output_img.scale(72 / DPI, 72 / DPI)
        output_img.drawOn(combined_pdf, 0, 0)
        # Complete page
        combined_pdf.showPage()

        if args.single_filename is not None:
            single_filename = args.single_filename.format(
                id=marker_id,
                marker_family=args.marker_family
            )
            renderPDF.drawToFile(output_img, single_filename)

    # Save combined PDF
    combined_pdf.save()


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """
    Marker_generator subparser TILE.

    Used to generate a PDF with multiple markers per page.
    """
    parser = subparsers.add_parser("TILE", help="Generate multiple markers per page")

    parser.add_argument(
        "--all_filename",
        type=str,
        help=(
            "Output filename of combined file. `id` and `marker_family` available for "
            "string format replacement (default: %(default)s)"
        ),
        default="combined_{marker_family}.pdf",
    )
    parser.add_argument(
        "--single_filename",
        type=str,
        help=(
            "Output filename of individual files. `id` and `marker_family` available for "
            "string format replacement"
        ),
        default=None,
    )
    parser.add_argument(
        "--page_size",
        type=str,
        help="Page size of output files (default: %(default)s)",
        choices=sorted([size.name for size in PageSize]),
        default="A4",
    )

    parser.add_argument(
        "--marker_family", default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--marker_size",
        help="The size of markers in millimeters (default: %(default)s)",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--aruco_orientation",
        help="Rotate marker 180 for aruco orientation",
        action="store_true",
    )
    parser.add_argument(
        "--range",
        help="Marker ids to output, can use '-' or ',' to specify lists and ranges",
        default="ALL",
    )

    parser.add_argument(
        "--no_number",
        help="Do not place id number on the marker",
        action="store_true",
    )
    parser.add_argument(
        "--number_size",
        help="Set the text size of the id number on the marker",
        default=DEFAULT_VEC_FONT_SIZE,
        type=int,
    )
    parser.add_argument(
        "--description_format",
        type=str,
        help=(
            "Text format for the description on the marker images. "
            "`marker_id` and `marker_type` are available for string format replacement. "
            "(default: '%(default)s')"
        ),
        default="{marker_type} {marker_id}",
    )
    parser.add_argument(
        "--description_size",
        help="Set the text size of the description text on the marker",
        default=DEFAULT_VEC_FONT_SIZE,
        type=int,
    )

    parser.add_argument(
        "--border_width",
        help="Size of the border in pixels (default: %(default)s)",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--border_colour",
        help="Colour of border elements (default: %(default)s)",
        default=DEFAULT_COLOUR,
        type=str,
    )
    parser.add_argument(
        "--tick_length",
        help="Length of center tick lines in pixels (default: %(default)s)",
        default=40,
        type=int,
    )

    lr_margin_group = parser.add_mutually_exclusive_group()
    lr_margin_group.add_argument(
        "--left_margin",
        help="Distance in mm between left border of marker and the left edge of the page",
        default=None,
        type=int,
    )
    lr_margin_group.add_argument(
        "--right_margin",
        help="Distance in mm between right border of marker and the right edge of the page",
        default=None,
        type=int,
    )

    tb_margin_group = parser.add_mutually_exclusive_group()
    tb_margin_group.add_argument(
        "--top_margin",
        help="Distance in mm between top border of marker and the top edge of the page",
        default=None,
        type=int,
    )
    tb_margin_group.add_argument(
        "--bottom_margin",
        help="Distance in mm between bottom border of marker and the bottom edge of the page",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--num_columns",
        help="Number of columns of markers to place",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--column_padding",
        help="Inner horizontal spacing between markers in mm",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--num_rows",
        help="Number of rows of markers to place",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--row_padding",
        help="Inner vertical spacing between markers in mm",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--repeat",
        help="Number of duplicates of each marker id to create",
        default=1,
        type=int,
    )

    parser.set_defaults(func=main)
