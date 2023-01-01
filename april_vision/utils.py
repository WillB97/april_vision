"""Generic utility functions."""
from collections import deque
from pathlib import Path
from typing import Deque, Tuple, Union

import cv2
import numpy as np

from .vision import Frame

Resolution = Tuple[int, int]
CameraIntrinsic = Tuple[float, float, float, float]


def annotate_text(
    frame: Frame,
    text: str,
    location: Tuple[int, int],
    text_scale: float = 0.5,
    text_colour: Tuple[int, int, int] = (255, 0, 0),
) -> Frame:
    """
    Add arbitrary text to a frame.

    text colour is in BGR format
    """
    for frame_type in frame:
        cv2.putText(
            frame_type,
            text,
            np.array(location, dtype=np.int32),
            cv2.FONT_HERSHEY_DUPLEX,
            text_scale,
            color=text_colour,  # in BGR
            thickness=2,
        )
    return frame


def load_calibration(calibration_file: Union[str, Path]) -> Tuple[Resolution, CameraIntrinsic]:
    """Load calibration data from opencv XML calibration file."""
    calibration_file = Path(calibration_file)

    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibrations not found: {calibration_file}")

    storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)

    resolution_node = storage.getNode("cameraResolution")
    width = int(resolution_node.at(0).real())
    height = int(resolution_node.at(1).real())

    camera_matrix = storage.getNode("cameraMatrix").mat()
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    return (
        (width, height),
        (fx, fy, cx, cy),
    )


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
