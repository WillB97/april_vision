"""Generic utility functions."""
from collections import deque
from math import hypot
from pathlib import Path
from typing import Deque, Dict, NamedTuple, Optional, Tuple, TypedDict, Union

import cv2
from numpy.typing import NDArray

from .calibrations import calibration_root
from .marker import Marker, PixelCoordinates


class Resolution(NamedTuple):
    """Resolution of the camera."""

    width: int
    height: int


class CameraIntrinsic(NamedTuple):
    """Camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float


class CameraCalibration(TypedDict):
    """Camera calibration data."""

    resolution: Resolution
    calibration: CameraIntrinsic
    vidpids: Tuple[str, ...]
    cameraProperties: Dict[int, int]


class Frame(NamedTuple):
    """A tuple of original image and a greyscale version for fiducial detection."""

    grey_frame: NDArray
    colour_frame: NDArray

    @classmethod
    def from_colour_frame(
        cls,
        colour_frame: NDArray,
        colourspace: Optional[int] = cv2.COLOR_BGR2GRAY,
    ) -> 'Frame':
        """Load frame from a colour image in a numpy array."""
        if colourspace is not None:
            grey_frame = cv2.cvtColor(colour_frame, colourspace)
        else:
            # mypy doesn't understand that Mat is a numpy array until numpy 2.1
            grey_frame = colour_frame.copy()  # type: ignore[assignment,unused-ignore]

        return cls(
            grey_frame=grey_frame,
            colour_frame=colour_frame,
        )

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'Frame':
        """Load an image file into the frame."""
        colour_frame = cv2.imread(str(filepath))
        if colour_frame is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")

        return cls.from_colour_frame(colour_frame)


def annotate_text(
    frame: Frame,
    text: str,
    location: Tuple[int, int],
    text_scale: float = 0.5,
    text_colour: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> Frame:
    """
    Add arbitrary text to a frame.

    text colour is in BGR format
    """
    for frame_type in frame:
        cv2.putText(
            frame_type,
            text,
            location,
            cv2.FONT_HERSHEY_DUPLEX,
            text_scale,
            color=text_colour,  # in BGR
            thickness=thickness,
        )
    return frame


def load_calibration_extra(calibration_file: Union[str, Path]) -> CameraCalibration:
    """Load full calibration data from opencv XML calibration file."""
    calibration_file = Path(calibration_file)

    if calibration_file.is_relative_to('__package__'):
        # ___package__ alias has been used, so we need to resolve the path
        calibration_file = calibration_root / calibration_file.relative_to('__package__')

    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibrations not found: {calibration_file}")

    storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)

    resolution_node = storage.getNode("cameraResolution")
    width = int(resolution_node.at(0).real())
    height = int(resolution_node.at(1).real())

    camera_matrix = storage.getNode("cameraMatrix").mat()
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    node = storage.getNode('vidpid')
    if node.isSeq():
        pidvids = tuple([node.at(i).string() for i in range(node.size())])
    elif node.isString():
        pidvids = tuple([node.string()])
    else:
        # This file lacks any vidpids
        pidvids = tuple()

    camera_props = {}
    cal_file_props = storage.getNode("cameraProperties").mat()
    if cal_file_props is not None:
        for property, value in cal_file_props:  # type: ignore[misc,unused-ignore]
            camera_props[property] = value

    storage.release()

    return CameraCalibration(
        resolution=Resolution(width, height),
        calibration=CameraIntrinsic(fx, fy, cx, cy),
        vidpids=pidvids,
        cameraProperties=camera_props,
    )


def load_calibration(calibration_file: Union[str, Path]) -> Tuple[Resolution, CameraIntrinsic]:
    """Load calibration data from opencv XML calibration file."""
    calibration_data = load_calibration_extra(calibration_file)

    return (
        calibration_data['resolution'],
        calibration_data['calibration'],
    )


def normalise_marker_text(marker: Marker) -> float:
    """
    Calculate text scale factor so that text on marker is a reasonable size.

    Based on the distance between corner diagonals.
    """
    corners = marker.pixel_corners

    def line_distance(p1: PixelCoordinates, p2: PixelCoordinates) -> float:
        return hypot(p1.x - p2.x, p1.y - p2.y)

    return max(
        line_distance(corners[0], corners[2]),
        line_distance(corners[1], corners[3]),
    ) / 300


class RollingAverage:
    """A rolling average filter using a deque."""

    def __init__(self, points: int) -> None:
        self.data: Deque[float] = deque(maxlen=points)
        self.points = points

    def new_data(self, value: float) -> None:
        """
        Add a new sample to the filter.

        Once the buffer is full, the oldest sample is dropped.
        """
        self.data.append(value)

    def average(self) -> float:
        """Return the average of the stored samples."""
        active_points = [data for data in self.data if data is not None]

        return sum(active_points) / len(active_points)
