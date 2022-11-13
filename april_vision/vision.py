from os import pathlike
from pathlib import Path
from typing import NamedTuple, Any, List, Tuple, Optional

import cv2
import numpy as np

from .marker import Marker

class Frame(NamedTuple):
    grey_frame: Any
    colour_frame: Any


def _find_camera():
    pass


# self._camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# self._camera.set(cv2.CAP_PROP_FPS, 30)
# self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class Camera:
    def __init__(
        self, 
        index: int, 
        resolution: Tuple[int, int], 
        calibration: Optional[Tuple[float, float, float, float]]=None
    ) -> None:
        self._camera = cv2.VideoCapture(index)
        self._set_resolution(resolution)
        self.calibraion = calibration

        # Take and discard a camera capture
        _ = self._capture_single_frame()

    @classmethod
    def from_calibration_file(cls, index: int, calibration_file: pathlike[str]) -> Camera:
        if not calibration_file.exists():
            raise FileNotFoundError(f"Calibrations not found: {calibration_file}")
        storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)
        resolution_node = storage.getNode("cameraResolution")
        camera_matrix = storage.getNode("cameraMatrix").mat()
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[3, 1], camera_matrix[3, 2]
        return cls(
            index,
            resolution=(
                int(resolution_node.at(0).real()),
                int(resolution_node.at(1).real()),
            ),
            calibration=(fx, fy, cx, cy),
        )

    def _set_camera_property(self, property: int, value: int) -> None:
        self._camera.set(property, value)
        actual = self._camera.get(property)

        assert actual == value, (f"Failed to set property '{property}', "
                                 f"expected {value} got {actual}")

    def _set_resolution(self, resolution: Tuple[int, int]) -> None:
        width, height = resolution
        self._set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._set_camera_property(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def _get_resolution(self) -> Tuple[int, int]:
        return (
            int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _capture_single_frame(self) -> np.ndarray:
        ret, colour_frame = self._camera.read()
        if not ret:
            raise IOError("Failed to get frama from camera")
        return colour_frame

    def _capture(self, fresh: bool=True) -> Frame:
        if fresh:
            _ = self._capture_single_frame()

        colour_frame = self._capture_single_frame()
        grey_frame = cv2.cvtColor(colour_frame, cv2.COLOR_BGR2GRAY)

        return Frame(grey_frame=grey_frame, colour_frame=colour_frame)

    def _detect(self, frame: Frame) -> List[Marker]:
        pass

    def _annotate(self, frame: Frame, corners: List[np.ndarray], ids: List[int]) -> Frame:
        pass

    def _save(self, frame: Frame, name: pathlike[str], colour: bool=True) -> None:
        if colour:
            output_frame = frame.colour_frame
        else:
            output_frame = frame.grey_frame

        path = Path(name)
        if not path.suffix:
            # TODO log we added and extension
            path = path.with_suffix(".jpg")

        cv2.imwrite(path, output_frame)

    def capture(self) -> np.ndarray:
        return self._capture().colour_frame

    def see(self) -> List[Marker]:
        frame = self._capture()
        return self._detect(frame)

    def see_ids(self) -> List[int]:
        frame = self._capture()
        markers = self._detect(frame)
        return [marker.id for marker in markers]

    def save(self, name):
        frame = self._capture()
        markers = self._detect(frame)
        frame = self._annotate(
            frame,
            corners=[marker.pixel_corners for marker in markers],
            ids=[marker.id for marker in markers])
        self._save(frame, name)

    def close(self):
        self._camera.release()
