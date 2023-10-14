import argparse
import logging

from PIL import Image

from april_vision.cli.utils import get_tag_family
from april_vision.marker import MarkerType

from ..marker_tile import MarkerTile
from ..utils import (DEFAULT_COLOUR, DEFAULT_FONT, DEFAULT_FONT_SIZE, DPI,
                     PageSize, mm_to_pixels, parse_marker_ranges)

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Generate a single marker on a page with the provided arguments"""
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    marker_ids = parse_marker_ranges(tag_data, args.range)

    page_size = PageSize[args.page_size]

    marker_pages = []

    for marker_id in marker_ids:
        LOGGER.info(f"Creating marker: {marker_id}")

        image_tile = MarkerTile(
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
                DEFAULT_FONT,
                args.number_size,
                args.border_colour,
            )

        image_tile.add_description_border(
            args.description_format,
            DEFAULT_FONT,
            args.description_size,
            "black",
            double_text=args.split,
        )

        if args.split:
            size = page_size.pixels[0] * 2, page_size.pixels[1]
            output_img = Image.new("RGB", size, (255, 255, 255))
        else:
            output_img = Image.new("RGB", page_size.pixels, (255, 255, 255))

        if args.left_margin is not None:
            x_loc = mm_to_pixels(args.left_margin) - image_tile.top_left.x
        elif args.right_margin is not None:
            x_loc = (output_img.width
                     - mm_to_pixels(args.right_margin)
                     - image_tile.bottom_right.x
                     )
        else:
            # Centred
            x_loc = (output_img.width - image_tile.image.width) // 2

        if args.top_margin is not None:
            y_loc = mm_to_pixels(args.top_margin) - image_tile.top_left.y
        elif args.bottom_margin is not None:
            y_loc = (output_img.height
                     - mm_to_pixels(args.bottom_margin)
                     - image_tile.bottom_right.y
                     )
        else:
            # Centred
            y_loc = (output_img.height - image_tile.image.height) // 2

        output_img.paste(image_tile.image, (x_loc, y_loc))

        if args.split:
            img_left = output_img.crop((
                0, 0,
                page_size.pixels[0] - 1, page_size.pixels[1] - 1
            ))
            img_right = output_img.crop((
                page_size.pixels[0], 0,
                (2 * page_size.pixels[0]) - 1, page_size.pixels[1] - 1
            ))
            marker_pages.append(img_left)
            marker_pages.append(img_right)
        else:
            marker_pages.append(output_img)

        if args.single_filename is not None:
            single_filename = args.single_filename.format(
                id=marker_id,
                marker_family=args.marker_family
            )
            if args.split:
                img_left.save(
                    f'left_{single_filename}',
                    quality=100,
                    dpi=(DPI, DPI),
                )
                img_right.save(
                    f'right_{single_filename}',
                    quality=100,
                    dpi=(DPI, DPI),
                )
            else:
                output_img.save(
                    single_filename,
                    quality=100,
                    dpi=(DPI, DPI),
                )

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
        choices=sorted([size.name for size in PageSize]),
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
        default=DEFAULT_FONT_SIZE,
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
        default=DEFAULT_FONT_SIZE,
        type=int,
    )

    parser.add_argument(
        "--border_width",
        help="Size of the border in pixels (default: %(default)s)",
        default=1,
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

    parser.set_defaults(func=main)
