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
            DEFAULT_COLOUR,
        )
        image_tile.add_centre_ticks(
            args.border_width,
            args.tick_length,
            DEFAULT_COLOUR,
        )

        if args.no_number is False:
            image_tile.add_id_number(
                args.font,
                args.number_size,
                DEFAULT_COLOUR,
            )

        image_tile.add_description_border(
            args.description_format,
            args.font,
            args.description_size,
            DEFAULT_COLOUR,
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
