import argparse
import logging

from PIL import Image

from april_vision.cli.utils import get_tag_family
from april_vision.marker import MarkerType

from ..marker_tile import MarkerTile
from ..utils import (DEFAULT_COLOUR, DPI, PageSize, mm_to_pixels,
                     parse_marker_ranges)

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    marker_ids = parse_marker_ranges(tag_data, args.range)

    marker_tiles = []

    for marker_id in marker_ids:
        image_tile = MarkerTile(
            tag_data,
            marker_id,
            args.marker_size,
            aruco_orientation=args.aruco_orientation
        )
        image_tile.add_border_line(
            args.border_width,
            DEFAULT_COLOUR
        )
        image_tile.add_centre_ticks(
            args.border_width,
            args.tick_length,
            DEFAULT_COLOUR
        )

        if args.no_number is False:
            image_tile.add_id_number(
                args.font,
                args.number_size,
                DEFAULT_COLOUR
            )

        image_tile.add_description_border(
            args.description_format,
            args.font,
            args.description_size,
            DEFAULT_COLOUR,
        )

        for i in range(args.repeat):
            marker_tiles.append(image_tile)

    page_size = PageSize[args.page_size]

    # Converts list of markers into a list of lists of markers per page
    markers_per_page = args.num_columns * args.num_rows
    marker_tiles_for_page = [
        marker_tiles[n:n + markers_per_page]
        for n in range(0, len(marker_tiles), markers_per_page)
    ]

    marker_pages = []
    for markers in marker_tiles_for_page:
        output_img = Image.new("RGB", page_size.pixels, (255, 255, 255))

        for index, marker in enumerate(markers):
            row, col = divmod(index, args.num_columns)

            if args.left_margin is not None:
                top_left_x = mm_to_pixels(args.left_margin)
            elif args.right_margin is not None:
                top_left_x = (output_img.width
                              - mm_to_pixels(args.right_margin)
                              - (args.num_columns * marker.marker_width)
                              - ((args.num_columns - 1) * mm_to_pixels(args.column_padding))
                              )
            else:
                # Centered
                top_left_x = (output_img.width
                              - (args.num_columns * marker.marker_width)
                              - ((args.num_columns - 1) * mm_to_pixels(args.column_padding))
                              ) // 2

            x_loc = (top_left_x
                     + (col * marker.marker_width)
                     + (col * mm_to_pixels(args.column_padding))
                     - marker.top_left.x
                     )

            if args.top_margin is not None:
                top_left_y = mm_to_pixels(args.top_margin)
            elif args.bottom_margin is not None:
                top_left_y = (output_img.height
                              - mm_to_pixels(args.bottom_margin)
                              - (args.num_rows * marker.marker_height)
                              - ((args.num_rows - 1) * mm_to_pixels(args.row_padding))
                              )
            else:
                # Centered
                top_left_y = (output_img.height
                              - (args.num_rows * marker.marker_height)
                              - ((args.num_rows - 1) * mm_to_pixels(args.row_padding))
                              ) // 2

            y_loc = (top_left_y
                     + (row * marker.marker_height)
                     + (row * mm_to_pixels(args.row_padding))
                     - marker.top_left.y
                     )

            output_img.paste(marker.image, (x_loc, y_loc))

        if args.single_filename is not None:
            single_filename = args.single_filename.format(
                id=marker_id,
                marker_family=args.marker_family
            )
            output_img.save(
                single_filename,
                quality=100,
                dpi=(DPI, DPI),
            )

        marker_pages.append(output_img)

    # Save combined PDF
    combined_filename = args.all_filename.format(
        marker_family=args.marker_family
    )

    first_page = marker_pages.pop(0)
    first_page.save(
        combined_filename,
        quality=100,
        dpi=(DPI, DPI),
        save_all=True,
        append_images=marker_pages,
    )


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("TILE")

    parser.add_argument(
        "--all_filename",
        type=str,
        help=(
            "Output filename of combined file. `id` available for string format replacement "
            "(default: %(default)s)"
        ),
        default="combined_{marker_family}.pdf",
    )
    parser.add_argument(
        "--single_filename",
        type=str,
        help=(
            "Output filename of split files. `id` available for string format replacement "
            "(default: %(default)s)"
        ),
        default=None,
    )
    parser.add_argument(
        "--page_size",
        type=str,
        help="Page size. (default: %(default)s)",
        choices=sorted([size.name for size in PageSize]),
        default="A4",
    )
    parser.add_argument(
        "--split",
        help="Split the marker image across two pages",
        action="store_true",
    )

    parser.add_argument(
        "--marker_family", default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'",
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
        help="Do not place marker id number on the marker",
        action="store_true",
    )
    parser.add_argument(
        "--number_size",
        help="Set the text size of the id number on the marker",
        default=12,
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
        default=12,
        type=int,
    )
    parser.add_argument(
        "--font",
        help="Set the text font (default: %(default)s)",
        default="calibri.ttf",
        type=str,
    )

    parser.add_argument(
        "--border_width",
        help="Size of the border in pixels (default: %(default)s)",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--tick_length",
        help="Length of center tick lines in pixels (default: %(default)s)",
        default=10,
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
        help="Inner horizontal spacing between markers",
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
        help="Inner vertical spacing between markers",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--repeat",
        help="Number of duplicates of each marker id number",
        default=1,
        type=int,
    )

    parser.set_defaults(func=main)
