"""
Generate marker PDFs.

Takes a set of parameters and generates marker PDFs.
"""
import argparse
import logging

from april_vision.marker import MarkerType

from .marker_main import main
from .utils import PageSize

LOGGER = logging.getLogger(__name__)

DPI = 72
BORDER_FILL = "lightgrey"


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_generator command parser."""

    parser = subparsers.add_parser(
        "marker_generator",
        description="Generate markers",
        help="Generate markers",
    )

    # Args for output filename/location
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

    # Args for modifying type and size
    # tag_group = parser.add_argument_group('TAGS')
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

    # Args for modifying marker design
    # text_group = parser.add_argument_group('TEXT')
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

    # Args for modifying grey border
    # border_group = parser.add_argument_group('STYLE')
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

    subparsers = parser.add_subparsers(required=True, help='mode', dest='mode')

    # Sub parser for single marker mode
    single_subparser = subparsers.add_parser("SINGLE")

    lr_single_margin_group = single_subparser.add_mutually_exclusive_group()
    lr_single_margin_group.add_argument(
        "--left_margin",
        help="Distance in mm between left border of marker and the left edge of the page",
        default=None,
        type=int,
    )
    lr_single_margin_group.add_argument(
        "--right_margin",
        help="Distance in mm between right border of marker and the right edge of the page",
        default=None,
        type=int,
    )

    tb_single_margin_group = single_subparser.add_mutually_exclusive_group()
    tb_single_margin_group.add_argument(
        "--top_margin",
        help="Distance in mm between top border of marker and the top edge of the page",
        default=None,
        type=int,
    )
    tb_single_margin_group.add_argument(
        "--bottom_margin",
        help="Distance in mm between bottom border of marker and the bottom edge of the page",
        default=None,
        type=int,
    )

    single_subparser.add_argument(
        "--split",
        help="Split the marker image across two pages",
        action="store_true",
    )

    # Sub parser for tile marker mode
    tile_subparser = subparsers.add_parser("TILE")

    lr_tile_margin_group = tile_subparser.add_mutually_exclusive_group()
    lr_tile_margin_group.add_argument(
        "--left_margin",
        help="Distance in mm between left border of marker and the left edge of the page",
        default=None,
        type=int,
    )
    lr_tile_margin_group.add_argument(
        "--right_margin",
        help="Distance in mm between right border of marker and the right edge of the page",
        default=None,
        type=int,
    )

    tb_tile_margin_group = tile_subparser.add_mutually_exclusive_group()
    tb_tile_margin_group.add_argument(
        "--top_margin",
        help="Distance in mm between top border of marker and the top edge of the page",
        default=None,
        type=int,
    )
    tb_tile_margin_group.add_argument(
        "--bottom_margin",
        help="Distance in mm between bottom border of marker and the bottom edge of the page",
        default=None,
        type=int,
    )

    tile_subparser.add_argument(
        "--num_columns",
        help="Number of columns of markers to place",
        default=1,
        type=int,
    )
    tile_subparser.add_argument(
        "--column_padding",
        help="Inner horizontal spacing between markers",
        default=0,
        type=int,
    )

    tile_subparser.add_argument(
        "--num_rows",
        help="Number of rows of markers to place",
        default=1,
        type=int,
    )
    tile_subparser.add_argument(
        "--row_padding",
        help="Inner vertical spacing between markers",
        default=0,
        type=int,
    )

    tile_subparser.add_argument(
        "--repeat",
        help="Number of duplicates of each marker id number",
        default=1,
        type=int,
    )

    # Image sub parser, doesnt use page dimention and provides image cropped to marker
    image_subparser = subparsers.add_parser("IMAGE")

    image_subparser.add_argument(
        "--border_size",
        help="Size of white border to add to the outside of the marker",
        default=0,
        type=int,
    )

    # Calibration board generator
    cal_subparser = subparsers.add_parser("CAL_BOARD")

    cal_subparser.add_argument(
        "--num_columns",
        help="Number of columns of markers to place",
        default=1,
        type=int,
    )
    cal_subparser.add_argument(
        "--num_rows",
        help="Number of rows of markers to place",
        default=1,
        type=int,
    )

    # Custom monolithic help
    # subparsers_actions = [
    #     action for action in parser._actions
    #     if isinstance(action, argparse._SubParsersAction)
    # ]
    # for subparsers_action in subparsers_actions:
    #     for choice, subparser in subparsers_action.choices.items():
    #         print("Mode: '{}'".format(choice))
    #         print(subparser.format_help())

    # def test(args):
    #    print(args)

    parser.set_defaults(func=main)
