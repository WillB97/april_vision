"""Utility functions for the marker generator."""
import logging
from enum import Enum
from typing import List, NamedTuple

from font_roboto import Roboto  # type: ignore[import,unused-ignore]
from reportlab.lib import colors as rl_colors

from april_vision.cli.utils import ApriltagFamily, parse_ranges

LOGGER = logging.getLogger(__name__)

DEFAULT_FONT = Roboto
DEFAULT_VEC_FONT = 'Times-Roman'
DEFAULT_FONT_SIZE = 50
DEFAULT_VEC_FONT_SIZE = 55
DEFAULT_COLOUR = "lightgrey"
VEC_DPI = 72
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
    """Convert millimeters to pixels."""
    inches = mm / 25.4
    return int(inches * DPI)


def mm_to_vec_pixels(mm: float) -> float:
    """Convert millimeters to pixels."""
    inches = mm / 25.4
    return inches * VEC_DPI


class Coord(NamedTuple):
    """Simple class to store coordinates."""

    x: int
    y: int


class VecCoord(NamedTuple):
    """
    Simple class to store coordinates.

    Floats are used to support defining vectors with subpixel accuracy.
    """

    x: float
    y: float


class PageSize(Enum):
    """Enum to define the dimentions of different page sizes."""

    A3 = (297, 420)
    A3L = (420, 297)
    A4 = (210, 297)
    A4L = (297, 210)

    @property
    def width(self) -> int:
        """Return the width of the page."""
        return self.value[0]

    @property
    def height(self) -> int:
        """Return the height of the page."""
        return self.value[1]

    @property
    def pixels(self) -> Coord:
        """Return the page size in pixels."""
        return Coord(
            mm_to_pixels(self.value[0]),
            mm_to_pixels(self.value[1]),
        )

    @property
    def vec_pixels(self) -> VecCoord:
        """Return the page size in pixels."""
        return VecCoord(
            mm_to_vec_pixels(self.value[0]),
            mm_to_vec_pixels(self.value[1]),
        )


class CustomPageSize:
    """Class to define a custom page size."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @property
    def pixels(self) -> Coord:
        """Return the custom page size in pixels."""
        return Coord(
            mm_to_pixels(self.width),
            mm_to_pixels(self.height),
        )

    @property
    def vec_pixels(self) -> VecCoord:
        """Return the page size in pixels."""
        return VecCoord(
            mm_to_vec_pixels(self.width),
            mm_to_vec_pixels(self.height),
        )


def get_reportlab_colour(col: str) -> rl_colors.Color:
    """Convert a string colour to a reportlab colour."""
    named_colours = rl_colors.getAllNamedColors()
    if col in named_colours:
        return named_colours[col]
    else:
        return rl_colors.HexColor(col)
