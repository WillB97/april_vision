import argparse
import logging

from april_vision.cli.utils import get_tag_family

from ..marker_tile import MarkerTile
from ..utils import DEFAULT_COLOUR, DPI, parse_marker_ranges

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    tag_data = get_tag_family(args.marker_family)
    LOGGER.info(tag_data)

    marker_ids = parse_marker_ranges(tag_data, args.range)

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

        image_tile.add_border_line(
            args.border_size,
            "white",
        )

        if args.single_filename is None:
            LOGGER.error("single_filename needs to be set")
            exit(1)

        single_filename = args.single_filename.format(
            id=marker_id,
            marker_family=args.marker_family
        )

        image_tile.image.save(
            single_filename,
            quality=100,
            dpi=(DPI, DPI),
        )
