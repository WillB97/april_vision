"""Marker_generator subparser SINGLE used to generate a PDF of a marker."""
import argparse
import logging
from typing import Union

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
    CustomPageSize,
    PageSize,
    mm_to_pixels,
    parse_marker_ranges,
)

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate a single marker on a page with the provided arguments."""
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    marker_ids = parse_marker_ranges(tag_data, args.range)

    page_size: Union[PageSize, CustomPageSize]
    if args.page_size == 'CROPPED':
        # Allow for an additional marker pixel border
        required_width = args.marker_size * (
            (tag_data.total_width + 2) / tag_data.width_at_border
        )
        page_size = CustomPageSize(required_width, required_width)
    else:
        page_size = PageSize[args.page_size]

    combined_filename = args.all_filename.format(
        marker_family=args.marker_family
    )
    combined_pdf = canvas.Canvas(combined_filename, pagesize=page_size.vec_pixels)

    if args.split:
        page_size = CustomPageSize(page_size.width * 2, page_size.height)

    for marker_id in marker_ids:
        LOGGER.info(f"Creating marker: {marker_id}")

        image_tile = MarkerTileVector(
            tag_data,
            marker_id,
            args.marker_size,
            aruco_orientation=args.aruco_orientation,
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
            double_text=args.split,
        )

        output_img = Drawing(page_size.vec_pixels.x, page_size.vec_pixels.y)

        # Default to center of the page
        x_loc, y_loc = page_size.pixels.x / 2, page_size.pixels.y / 2

        # Use center of marker as reference point
        if args.left_margin is not None:
            x_loc = mm_to_pixels(args.left_margin) + image_tile.marker_width / 2
        elif args.right_margin is not None:
            x_loc = (
                page_size.pixels.x
                - mm_to_pixels(args.right_margin)
                - image_tile.marker_width / 2
            )

        if args.top_margin is not None:
            y_loc = (
                page_size.pixels.y
                - mm_to_pixels(args.top_margin)
                - image_tile.marker_width / 2
            )
        elif args.bottom_margin is not None:
            y_loc = mm_to_pixels(args.bottom_margin) + image_tile.marker_width / 2

        image_tile.set_marker_centre(x_loc, y_loc)

        if args.split:
            img_left = Drawing(page_size.vec_pixels.x / 2, page_size.vec_pixels.y)
            img_right = Drawing(page_size.vec_pixels.x / 2, page_size.vec_pixels.y)

            img_left.add(image_tile.vectors)
            # Translate to the right half of the page
            right_vectors = image_tile.vectors.copy()
            right_vectors.translate(-page_size.pixels.x / 2, 0)
            img_right.add(right_vectors)
            # canvas DPI is 72
            img_left.scale(72 / DPI, 72 / DPI)
            img_right.scale(72 / DPI, 72 / DPI)

            img_left.drawOn(combined_pdf, 0, 0)
            # Complete page
            combined_pdf.showPage()

            img_right.drawOn(combined_pdf, 0, 0)
            # Complete page
            combined_pdf.showPage()
        else:
            output_img.add(image_tile.vectors)
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
            if args.split:
                renderPDF.drawToFile(img_left, f'left_{single_filename}')
                renderPDF.drawToFile(img_right, f'right_{single_filename}')
            else:
                renderPDF.drawToFile(output_img, single_filename)

    # Save combined PDF
    combined_pdf.save()


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_generator subparser SINGLE used to generate a PDF of a marker."""
    parser = subparsers.add_parser("SINGLE", help="Generate a single marker per page")

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
        choices=sorted([size.name for size in PageSize] + ["CROPPED"]),
        default="A4",
    )
    parser.add_argument(
        "--split",
        help="Split the marker image across two pages",
        action="store_true",
    )

    parser.add_argument(
        "--marker_family",
        default=MarkerType.APRILTAG_36H11.value,
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

    parser.set_defaults(func=main)
