import logging
from enum import Enum
from typing import List, NamedTuple, Tuple

from april_vision.cli.utils import ApriltagFamily, parse_ranges

LOGGER = logging.getLogger(__name__)

DEFAULT_COLOUR = "lightgrey"
DPI = 72


class coord(NamedTuple):
    x: int
    y: int


def parse_marker_ranges(marker_family: ApriltagFamily, range_str: str) -> List[int]:
    # Get list of markers we want to make
    if range_str == "ALL":
        marker_ids = [num for num in range(marker_family.ncodes)]
    else:
        try:
            marker_ids = parse_ranges(range_str)
        except ValueError:
            LOGGER.error("Invalid marker number range provided")
            exit(1)

    if (max(marker_ids) > (marker_family.ncodes - 1)) or (min(marker_ids) < 0):
        LOGGER.error("Supplied marker number lies outside permitted values for marker family")
        LOGGER.error(f"Permitted marker range: 0-{marker_family.ncodes - 1}")
        exit(1)

    return marker_ids


def mm_to_pixels(mm: float) -> int:
    """
    Convert millimeters to pixels
    """
    inches = mm / 25.4
    return int(inches * DPI)


class PageSize(Enum):
    A3 = (297, 420)
    A3L = (420, 297)
    A4 = (210, 297)
    A4L = (297, 210)

    @property
    def pixels(self) -> Tuple[int, int]:
        return (
            mm_to_pixels(self.value[0]),
            mm_to_pixels(self.value[1]),
        )
