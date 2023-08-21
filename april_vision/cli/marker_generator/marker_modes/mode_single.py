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

    # try:
    #     output_img.save(
    #         output_dir / args.filename.format(id=marker_id),
    #         quality=100,
    #         dpi=(DPI, DPI),
    #     )
    # except ValueError as error:
    #     LOGGER.error(f"Invalid output file format: '{args.filename}'")
    #     LOGGER.error(error)
    #     exit(1)

    # # Output page size and location
    # page_size = PageSize[args.page_size]

    # output_dir = args.output_dir.resolve()
    # output_dir.mkdir(exist_ok=True, parents=True)

    # if args.merge_pdf is not None:
    #     if args.filename.lower().endswith(".pdf"):
    #         LOGGER.info("Starting to merge PDFs")
    #         merger = PdfMerger()
    #         for pdf in generated_files:
    #             merger.append(pdf)

    #         if args.merge_pdf.lower().endswith(".pdf"):
    #             merger.write(output_dir / args.merge_pdf)
    #         else:
    #             merger.write(output_dir / (args.merge_pdf + '.pdf'))

    #         merger.close()
    #         LOGGER.info("Merge PDF complete")
    #     else:
    #         LOGGER.error((
    #             "PDF merge was enabled but no PDFs were generated, "
    #             f"output format '{args.filename}'"
    #         ))
