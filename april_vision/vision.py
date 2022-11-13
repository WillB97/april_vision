from os import pathlike
from pathlib import Path
from typing import NamedTuple, Any, List

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

class USBCamera:
    def __init__(self, resolution, calibration):
        self._camera = cv2.VideoCapture(0)  # TODO Detect this better

        # Take and discard a camera capture
        _ = self._camera.read()

    def _set_camera_property(self, propery, value):
        self._camera.set(propery, value)
        actual = self._camera.get(propery)

        assert actual == value, (f"Failed to set property '{propery}', "
                                 f"expected {value} got {actual}")

    def set_resolution(self, resolution):
        width, height = resolution
        self._set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._set_camera_property(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_resolution(self):
        return (
            int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def capture(self):
        _ = self._camera.read()
        ret, colour_frame = self._camera.read()

        if not ret:
            raise IOError("Capture from camera failed")

        grey_frame = cv2.cvtColor(colour_frame, cv2.COLOR_BGR2GRAY)

        return Frame(
            grey_frame=grey_frame,
            colour_frame=colour_frame
        )

    def close(self):
        self._camera.release()


class Camera:
    def __init__(self, index: int) -> None:
        self.video_device = cv2.VideoCapture(index)
        pass

    def _capture_single_frame(self) -> np.ndarray:
        ret, colour_frame = self.video_device.read()
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
