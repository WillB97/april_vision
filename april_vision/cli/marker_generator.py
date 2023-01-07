"""
Generate marker PDFs.

Takes a set of parameters and generates marker PDFs.
"""
import argparse
import logging
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np
import pyapriltags
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..marker import MarkerType

LOGGER = logging.getLogger(__name__)

DPI = 72
BORDER_FILL = "lightgrey"


def mm_to_inches(mm: int) -> float:
    """
    Convert millimeters to inches
    """
    inches = mm / 25.4
    inches = round(inches, 4)
    return inches


def mm_to_pixels(mm: int) -> int:
    return int(mm_to_inches(mm) * DPI)


class PageSize(Enum):
    A3 = (297, 420)
    A4 = (210, 297)

    @property
    def pixels(self) -> tuple[int, int]:
        return (
            mm_to_pixels(self.value[0]),
            mm_to_pixels(self.value[1]),
        )


class ApriltagFamily(NamedTuple):
    ncodes: int
    codes: List[int]
    width_at_border: int
    total_width: int
    reversed_border: bool
    nbits: int
    bits: List[Tuple[int, int]]
    h: int
    name: str


def get_tag_family(family: str) -> ApriltagFamily:
    d = pyapriltags.Detector(families=family)
    raw_tag_data = d.tag_families[family].contents

    tag_data = ApriltagFamily(
        ncodes=raw_tag_data.ncodes,
        codes=[raw_tag_data.codes[i] for i in range(raw_tag_data.ncodes)],
        width_at_border=raw_tag_data.width_at_border,
        total_width=raw_tag_data.total_width,
        reversed_border=raw_tag_data.reversed_border,
        nbits=raw_tag_data.nbits,
        bits=[
            (raw_tag_data.bit_x[i], raw_tag_data.bit_y[i])
            for i in range(raw_tag_data.nbits)
        ],
        h=raw_tag_data.h,
        name=raw_tag_data.name.decode("utf-8"),
    )
    return tag_data


def parse_ranges(ranges: str) -> set[int]:
    """
    Parse a comma seprated list of numbers which may include ranges
    specified as hyphen-separated numbers.
    From https://stackoverflow.com/questions/6405208
    """
    result: list[int] = []
    for part in ranges.split(","):
        if "-" in part:
            a_, b_ = part.split("-")
            a, b = int(a_), int(b_)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return set(result)


def generate_tag_array(tag_data: ApriltagFamily, tag_id: int) -> np.ndarray:
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
            font=ImageFont.truetype("NotoSans-Regular.ttf", args.number_size),
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
        font=ImageFont.truetype("NotoSans-Regular.ttf", marker_sqaure_size // 2),
    )

    return text_border_image


def main(args: argparse.Namespace):
    """Annotate an image file using the provided args."""
    tag_data = get_tag_family(args.tag_family)

    # Get list of markers we want to make
    if args.range == "ALL":
        marker_ids = [num for num in range(tag_data.ncodes)]
    else:
        marker_ids = list(parse_ranges(args.range))

    # Output page size and location
    page_size = PageSize[args.page_size]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate markers
    for marker_id in marker_ids:
        image_tile = generate_tag_tile(tag_data, args, tag_id=marker_id)
        paper_img = Image.new("RGB", page_size.pixels, (255, 255, 255))
        paper_img.paste(
            image_tile,
            (
                (page_size.pixels[0] - image_tile.size[0]) // 2,
                (page_size.pixels[1] - image_tile.size[0]) // 2,
            ),
        )
        paper_img.save(
            output_dir / args.filename.format(id=marker_id),
            quality=100,
            dpi=(DPI, DPI),
        )


def create_subparser(subparsers: argparse._SubParsersAction):
    """Marker_generator command parser."""
    parser = subparsers.add_parser(
        "marker_generator",
        description="Generate a PDF containing markers",
        help="Generate a PDF containing markers",
    )

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
        "--range",
        help="Marker ids to output, can use '-' or ',' to specify lists and ranges",
        default="ALL",
    )

    parser.add_argument(
        "--page_size",
        type=str,
        help="Page size. (default: %(default)s)",
        choices=sorted([size.name for size in PageSize]),
        default="A4",
    )

    # parser.add_argument(
    #     "--force-a4",
    #     help="Output the PDF onto A4, splitting as necessary",
    #     action="store_true",
    # )

    parser.add_argument(
        "--no_number",
        help="Do not place marker id number on the marker",
        action="store_true",
    )
    parser.add_argument(
        "--number_size",
        help="Set the text size of the id number on the marker",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--description_format",
        type=str,
        help=(
            "Text format for the description on the marker images. "
            "`marker_id` and `marker_type` are available for string format replacement. "
            "(default: %(default)s)"
        ),
        default="{marker_type} {marker_id}",
    )

    parser.add_argument(
        "--tag_family", default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")

    parser.add_argument(
        "--tag_size",
        help="The size of markers in millimeters (default: %(default)s)",
        default=100,
        type=int,
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

    parser.add_argument(
        "--aruco_orientation",
        help="Rotate marker 180 for aruco orientation",
        action="store_true",
    )

    parser.set_defaults(func=main)
