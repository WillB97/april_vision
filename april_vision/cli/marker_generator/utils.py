import logging
from enum import Enum
from typing import List, Tuple

from font_roboto import Roboto  # type: ignore[import]

from april_vision.cli.utils import ApriltagFamily, parse_ranges

LOGGER = logging.getLogger(__name__)

DEFAULT_FONT = Roboto
DEFAULT_FONT_SIZE = 50
DEFAULT_COLOUR = "lightgrey"
DPI = 300


def parse_marker_ranges(marker_family: ApriltagFamily, range_str: str) -> List[int]:
    """
    Utility function to parse the provided range of markers.
    Also checks bounds against the marker family.
    """
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
    """
    Enum to define the dimentions of different page sizes
    """
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


class CustomPageSize:
    """
    Class to define a custom page size
    """
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @property
    def pixels(self) -> Tuple[int, int]:
        return (
            mm_to_pixels(self.width),
            mm_to_pixels(self.height),
        )
