"""
Generate marker PDFs.

Takes a set of parameters and generates marker PDFs.
"""
import argparse
import logging
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pypdf import PdfMerger

from ..marker import MarkerType
from .utils import ApriltagFamily, get_tag_family, parse_ranges

LOGGER = logging.getLogger(__name__)

DPI = 72
BORDER_FILL = "lightgrey"


def mm_to_pixels(mm: int) -> int:
    """
    Convert millimeters to pixels
    """
    inches = mm / 25.4
    return int(inches * DPI)


class PageSize(Enum):
    A3 = (297, 420)
    A4 = (210, 297)

    @property
    def pixels(self) -> Tuple[int, int]:
        return (
            mm_to_pixels(self.value[0]),
            mm_to_pixels(self.value[1]),
        )


class PageMode(Enum):
    SINGLE = 1
    TILE = 2
    CROP = 3
    SPLIT = 4


def generate_tag_array(tag_data: ApriltagFamily, tag_id: int) -> NDArray:
    """
    Uses the tag family object to generate a marker, returns this data as a 2d numpy array
    where each of the cells is 1 pixel of the marker.
    """
    # Create grid of tag size
    dim = tag_data.total_width
    tag = np.ones((dim, dim), dtype=np.uint8) * 255

    # Find position of origin in grid
    border_edge = int((tag_data.total_width / 2.0) - (tag_data.width_at_border / 2.0))

    # Draw black/white boarder lines
    if tag_data.reversed_border:
        left_border_edge = border_edge - 1
        right_border_edge = border_edge + tag_data.width_at_border
    else:
        left_border_edge = border_edge
        right_border_edge = border_edge + tag_data.width_at_border - 1

    for i in range(left_border_edge, right_border_edge+1):
        tag[left_border_edge][i] = 0
        tag[i][left_border_edge] = 0
        tag[right_border_edge][i] = 0
        tag[i][right_border_edge] = 0

    # Fill in pixels, black if bit index is zero
    max_index = tag_data.nbits - 1
    for i, (x, y) in enumerate(tag_data.bits):
        binary = bool(tag_data.codes[tag_id] & (1 << (max_index - i)))
        tag[y+border_edge][x+border_edge] = 255 if binary else 0

    return tag


def generate_tag_tile(
    tag_data: ApriltagFamily,
    args: argparse.Namespace,
    tag_id: int,
) -> Image.Image:
    """
    Uses the tag family object to generate a marker image.
    The marker is scaled to the correct size and is annotated with a border and text.
    """
    # Calculate the overall marker size
    pixel_size = args.tag_size // tag_data.width_at_border
    required_size = pixel_size * tag_data.total_width

    # Generate marker image and resize
    tag_array = generate_tag_array(tag_data, tag_id)

    if args.aruco_orientation:
        # Rotate by 180deg to match the aruco format
        tag_array = np.rot90(tag_array, k=2)

    marker_image = Image.fromarray(tag_array)
    resized_image = marker_image.resize(
        (mm_to_pixels(required_size), mm_to_pixels(required_size)),
        resample=0
    )

    # Add grey border line
    bordered_image = ImageOps.expand(
        resized_image,
        border=args.border_width,
        fill=BORDER_FILL
    )
    img_size = bordered_image.size[0]
    image_draw = ImageDraw.Draw(bordered_image)

    # Add center tick marks
    line_start = (img_size // 2) - (args.border_width // 2)

    # Top
    image_draw.line(
        [line_start, 0, line_start, args.tick_length],
        width=args.border_width,
        fill=BORDER_FILL,
    )

    # Left
    image_draw.line(
        [0, line_start, args.tick_length, line_start],
        width=args.border_width,
        fill=BORDER_FILL,
    )

    # Bottom
    image_draw.line(
        [line_start, img_size - args.tick_length, line_start, img_size],
        width=args.border_width,
        fill=BORDER_FILL,
    )

    # Right
    image_draw.line(
        [img_size - args.tick_length, line_start, img_size, line_start],
        width=args.border_width,
        fill=BORDER_FILL,
    )

    # Add text to the image
    marker_sqaure_size = mm_to_pixels(pixel_size)

    # Draw tag ID number in corner of white boarder
    border_edge = int((tag_data.total_width / 2.0) - (tag_data.width_at_border / 2.0))
    if tag_data.reversed_border:
        id_pos = (marker_sqaure_size * border_edge) + (marker_sqaure_size // 2)
    else:
        id_pos = (marker_sqaure_size * (border_edge - 1)) + (marker_sqaure_size // 2)

    if args.no_number is False:
        image_draw.text(
            (id_pos, id_pos),
            f"{tag_id}",
            fill=BORDER_FILL,
            anchor="mm",
            font=ImageFont.truetype(args.font, args.number_size),
        )

    # Expand the tile to add a white border with a width of 1 marker square
    text_border_image = ImageOps.expand(
        bordered_image,
        border=marker_sqaure_size,
        fill="white"
    )
    image_draw_expand = ImageDraw.Draw(text_border_image)

    # Draw text outside the marker
    image_draw_expand.text(
        (marker_sqaure_size, text_border_image.size[0] - (marker_sqaure_size // 2)),
        args.description_format.format(
            marker_type=tag_data.name, marker_id=tag_id
        ),
        anchor="lm",
        font=ImageFont.truetype(args.font, args.number_size),
    )

    return text_border_image


def main(args: argparse.Namespace) -> None:
    """
    Create markers and marker PDFs using the parameters provided on the command line.
    """
    tag_data = get_tag_family(args.tag_family)
    LOGGER.info(tag_data)

    # Get list of markers we want to make
    if args.range == "ALL":
        marker_ids = [num for num in range(tag_data.ncodes)]
    else:
        try:
            marker_ids = parse_ranges(args.range)
        except ValueError:
            LOGGER.error("Invalid marker number range provided")
            exit(1)

    if (max(marker_ids) > (tag_data.ncodes - 1)) or (min(marker_ids) < 0):
        LOGGER.error("Supplied marker number lies outside permitted values for marker family")
        LOGGER.error(f"Permitted marker range: 0-{tag_data.ncodes - 1}")
        exit(1)

    LOGGER.info(f"Generating {len(marker_ids)} markers")

    # Output page size and location
    page_size = PageSize[args.page_size]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    generated_files = []

    # Generate markers
    for marker_id in marker_ids:
        LOGGER.info(f"Creating marker: {marker_id}")

        image_tile = generate_tag_tile(tag_data, args, tag_id=marker_id)
        tile_size = image_tile.size[0]

        page_mode = PageMode[args.page_mode]

        if page_mode == PageMode.SINGLE:
            output_img = Image.new("RGB", page_size.pixels, (255, 255, 255))

            if args.bottom_padding is None:
                x_loc = (page_size.pixels[0] - tile_size) // 2
                y_loc = (page_size.pixels[1] - tile_size) // 2
                output_img.paste(image_tile, (x_loc, y_loc))
            else:
                x_loc = (page_size.pixels[0] - tile_size) // 2

                # Awkward math as we extended the tile beyond the border as extra area for
                # the text, The size of the image tile does not match the border
                marker_sqaure_size = mm_to_pixels(args.tag_size // tag_data.width_at_border)
                y_loc = (page_size.pixels[1]
                         - mm_to_pixels(args.bottom_padding)
                         - (marker_sqaure_size * (tag_data.total_width + 1))
                         - args.border_width
                         )
                output_img.paste(image_tile, (x_loc, y_loc))

            generated_files.append(output_dir / args.filename.format(id=marker_id))
            try:
                output_img.save(
                    output_dir / args.filename.format(id=marker_id),
                    quality=100,
                    dpi=(DPI, DPI),
                )
            except ValueError as error:
                LOGGER.error(f"Invalid output file format: '{args.filename}'")
                LOGGER.error(error)
                exit(1)

        elif page_mode == PageMode.TILE:
            output_img = Image.new("RGB", page_size.pixels, (255, 255, 255))

            for row in range(args.row_num):
                for col in range(args.column_num):
                    col_space_px = mm_to_pixels(args.column_spacing)
                    row_space_px = mm_to_pixels(args.row_spacing)
                    x_loc = ((col_space_px + tile_size) * col) + col_space_px
                    y_loc = ((row_space_px + tile_size) * row) + row_space_px

                    output_img.paste(image_tile, (x_loc, y_loc))

            generated_files.append(output_dir / args.filename.format(id=marker_id))
            try:
                output_img.save(
                    output_dir / args.filename.format(id=marker_id),
                    quality=100,
                    dpi=(DPI, DPI),
                )
            except ValueError as error:
                LOGGER.error(f"Invalid output file format: '{args.filename}'")
                LOGGER.error(error)
                exit(1)

        elif page_mode == PageMode.CROP:
            if args.crop_size is None:
                pixel_size = args.tag_size // tag_data.width_at_border
                required_size = pixel_size * tag_data.total_width
                crop_pixels = mm_to_pixels(required_size) + args.border_width
            else:
                crop_pixels = mm_to_pixels(args.crop_size)

            output_img = Image.new("RGB", (crop_pixels, crop_pixels), (255, 255, 255))
            x_loc = (crop_pixels - tile_size) // 2
            y_loc = (crop_pixels - tile_size) // 2
            output_img.paste(image_tile, (x_loc, y_loc))

            try:
                output_img.save(
                    output_dir / args.filename.format(id=marker_id),
                    quality=100,
                    dpi=(DPI, DPI),
                )
            except ValueError as error:
                LOGGER.error(f"Invalid output file format: '{args.filename}'")
                LOGGER.error(error)
                exit(1)

        elif page_mode == PageMode.SPLIT:
            output_img_1 = Image.new("RGB", page_size.pixels, (255, 255, 255))
            output_img_2 = Image.new("RGB", page_size.pixels, (255, 255, 255))

            # Draw text outside the marker in top right corner
            marker_sqaure_size = mm_to_pixels(args.tag_size // tag_data.width_at_border)

            image_draw_text = ImageDraw.Draw(image_tile)
            image_draw_text.text(
                (tile_size - marker_sqaure_size, (marker_sqaure_size // 2)),
                args.description_format.format(
                    marker_type=tag_data.name, marker_id=marker_id
                ),
                anchor="rm",
                font=ImageFont.truetype(args.font, args.number_size),
            )

            image_half_left = image_tile.crop((0, 0, tile_size // 2, tile_size))
            image_half_right = image_tile.crop((tile_size // 2, 0, tile_size, tile_size))

            x_loc = (page_size.pixels[0] - image_half_left.size[0]) // 2
            y_loc = (page_size.pixels[1] - image_half_left.size[1]) // 2

            output_img_1.paste(image_half_left, (x_loc, y_loc))
            output_img_2.paste(image_half_right, (x_loc, y_loc))

            generated_files.append(output_dir / args.filename.format(id=marker_id))
            try:
                output_img_1.save(
                    output_dir / args.filename.format(id=marker_id),
                    quality=100,
                    dpi=(DPI, DPI),
                    save_all=True,
                    append_images=[
                        output_img_2
                    ],
                )
            except ValueError as error:
                LOGGER.error(f"Invalid output file format: '{args.filename}'")
                LOGGER.error(error)
                exit(1)

    if args.merge_pdf is not None:
        if args.filename.lower().endswith(".pdf"):
            LOGGER.info("Starting to merge PDFs")
            merger = PdfMerger()
            for pdf in generated_files:
                merger.append(pdf)

            if args.merge_pdf.lower().endswith(".pdf"):
                merger.write(output_dir / args.merge_pdf)
            else:
                merger.write(output_dir / (args.merge_pdf + '.pdf'))

            merger.close()
            LOGGER.info("Merge PDF complete")
        else:
            LOGGER.error((
                "PDF merge was enabled but no PDFs were generated, "
                f"output format '{args.filename}'"
            ))


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_generator command parser."""

    parser = subparsers.add_parser(
        "marker_generator",
        description="Generate a PDF containing markers",
        help="Generate a PDF containing markers",
    )

    # Args for output filename/location
    parser.add_argument(
        "output_dir",
        help="The directory to save the output files to",
        type=Path,
    )
    parser.add_argument(
        "--filename",
        type=str,
        help=(
            "Output filename. `id` available for string format replacement "
            "(default: %(default)s)"
        ),
        default="{id}.pdf",
    )
    parser.add_argument(
        "--merge_pdf",
        help="Merge all the outputted PDFs into a single PDF of the provided filename",
        type=str,
        default=None,
    )

    # Args for modifying type and size
    tag_group = parser.add_argument_group('TAGS')
    tag_group.add_argument(
        "--tag_family", default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'",
    )
    tag_group.add_argument(
        "--tag_size",
        help="The size of markers in millimeters (default: %(default)s)",
        default=100,
        type=int,
    )
    tag_group.add_argument(
        "--no_aruco_orientation",
        help="Rotate marker 180 for standard orientation",
        action="store_false",
        dest="aruco_orientation",
    )
    tag_group.add_argument(
        "--range",
        help="Marker ids to output, can use '-' or ',' to specify lists and ranges",
        default="ALL",
    )

    # Args for modifying page size and marker layout
    page_group = parser.add_argument_group('PAGE')
    page_group.add_argument(
        "--page_size",
        type=str,
        help="Page size. (default: %(default)s)",
        choices=sorted([size.name for size in PageSize]),
        default="A4",
    )
    page_group.add_argument(
        "--page_mode",
        type=str,
        help="Page generation method. (default: %(default)s)",
        choices=sorted([mode.name for mode in PageMode]),
        default="SINGLE",
    )

    # Options for a SINGLE layout
    single_group = parser.add_argument_group('mode: SINGLE')
    single_group.add_argument(
        "--bottom_padding",
        help="Distance in mm between bottom border of marker and the bottom of the page",
        default=None,
        type=int,
    )

    # Options for a TILE layout
    tile_group = parser.add_argument_group('mode: TILE')
    tile_group.add_argument(
        "--column_num",
        help="Set number of columns of markers",
        default=1,
        type=int,
    )
    tile_group.add_argument(
        "--column_spacing",
        help="Set spacing between columns of markers",
        default=0,
        type=int,
    )
    tile_group.add_argument(
        "--row_num",
        help="Set number of rows of markers",
        default=1,
        type=int,
    )
    tile_group.add_argument(
        "--row_spacing",
        help="Set number of rows of markers",
        default=0,
        type=int,
    )

    # Options for CROP layout
    crop_group = parser.add_argument_group('mode: CROP')
    crop_group.add_argument(
        "--crop_size",
        help=(
            "Set dimension in mm of the cropped image in a CROP page mode, "
            "defaults to the marker border"
        ),
        default=None,
        type=int,
    )

    # Args for modifying text
    text_group = parser.add_argument_group('TEXT')
    text_group.add_argument(
        "--no_number",
        help="Do not place marker id number on the marker",
        action="store_true",
    )
    text_group.add_argument(
        "--number_size",
        help="Set the text size of the id number on the marker",
        default=10,
        type=int,
    )
    text_group.add_argument(
        "--description_format",
        type=str,
        help=(
            "Text format for the description on the marker images. "
            "`marker_id` and `marker_type` are available for string format replacement. "
            "(default: %(default)s)"
        ),
        default="{marker_type} {marker_id}",
    )
    text_group.add_argument(
        "--font",
        help="Set the text font (default: %(default)s)",
        default="calibri.ttf",
        type=str,
    )

    # Args for modifying grey border
    border_group = parser.add_argument_group('STYLE')
    border_group.add_argument(
        "--border_width",
        help="Size of the border in pixels (default: %(default)s)",
        default=1,
        type=int,
    )
    border_group.add_argument(
        "--tick_length",
        help="Length of center tick lines in pixels (default: %(default)s)",
        default=10,
        type=int,
    )

    parser.set_defaults(func=main)
