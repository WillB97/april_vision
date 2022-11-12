import cv2
from typing import NamedTuple, Any


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

