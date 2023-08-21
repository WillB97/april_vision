import argparse
import logging

from PIL import Image

from april_vision.cli.utils import get_tag_family

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
                              - ((args.num_columns - 1) * args.column_padding)
                              )
            else:
                # Centered
                top_left_x = (output_img.width
                              - (args.num_columns * marker.marker_width)
                              - ((args.num_columns - 1) * args.column_padding)
                              ) // 2

            x_loc = (top_left_x
                     + (col * marker.marker_width)
                     + (col * args.column_padding)
                     - marker.top_left.x
                     )

            if args.top_margin is not None:
                top_left_y = mm_to_pixels(args.top_margin)
            elif args.bottom_margin is not None:
                top_left_y = (output_img.height
                              - mm_to_pixels(args.bottom_margin)
                              - (args.num_rows * marker.marker_height)
                              - ((args.num_rows - 1) * args.row_padding)
                              )
            else:
                # Centered
                top_left_y = (output_img.height
                              - (args.num_rows * marker.marker_height)
                              - ((args.num_rows - 1) * args.row_padding)
                              ) // 2

            y_loc = (top_left_y
                     + (row * marker.marker_height)
                     + (row * args.row_padding)
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
