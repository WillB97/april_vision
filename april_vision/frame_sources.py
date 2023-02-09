"""The available frame sources for the Processor class."""
import logging
from pathlib import Path
from sys import platform
from typing import List, Optional, Tuple, Union

import cv2
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


class FrameSource:
    """
    A base frame source.

    Allows april_vision.Processor to be created prior to frames being available.
    """

    def read(self, fresh: bool = True) -> NDArray:
        """
        The method for getting a new frame.

        :param fresh: Whether to flush the device's buffer before capturing
        the frame, unused in the base class.
        """
        raise NotImplementedError("This frame source does not produce any frames.")

    def close(self) -> None:
        """Default method for closing the underlying capture device."""
        pass


class USBCamera(FrameSource):
    """A USB attached camera."""

    def __init__(
        self,
        index: int,
        resolution: Tuple[int, int] = (0, 0),
        camera_parameters: Optional[List[Tuple[int, int]]] = None,
        calibration: Optional[Tuple[float, float, float, float]] = None,
        vidpid: str = "",
    ) -> None:
        """
        Create a USB attached camera frame source.

        :param index: The camera's opencv index.
        :param resolution: Resolution to set the camera to.
        :param camera_parameters: Additional opencv parameters to apply to the camera.
        :param calibration: Optional, the intrinsic parameters of the camera.
        :param vidpid: The vendor/product string for the camera.
        """
        self._camera = cv2.VideoCapture(index)

        if resolution != (0, 0):
            try:
                self._set_resolution(resolution)
            except AssertionError as e:
                LOGGER.warning(f"Failed to set resolution: {e}")

        if camera_parameters is None:
            camera_parameters = []

        # Set buffer length to 1
        camera_parameters.append((cv2.CAP_PROP_BUFFERSIZE, 1))

        for parameter, value in camera_parameters:
            try:
                self._set_camera_property(parameter, value)
            except AssertionError as e:
                LOGGER.warning(f"Failed to set property: {e}")

        # Maximise the framerate on Linux
        # NOTE optimisation is disabled currently due to
        # 'Corrupt JPEG data: 5 extraneous bytes' seen on raspberry pi 4's
        # self._optimise_camera(vidpid)

        self._buffer_length = int(self._camera.get(cv2.CAP_PROP_BUFFERSIZE))

        # Take and discard a camera capture
        _ = self._capture_single_frame()

        # Save the camera intrinsic parameters
        self.calibration = calibration

    @classmethod
    def from_calibration_file(
        cls,
        index: int,
        calibration_file: Union[str, Path, None],
        vidpid: str = "",
    ) -> 'USBCamera':
        """Instantiate camera using calibration data from opencv XML calibration file."""
        if calibration_file is not None:
            calibration_file = Path(calibration_file)
        else:
            return cls(index)

        if not calibration_file.exists():
            LOGGER.warning(f"Calibrations not found: {calibration_file}")
            return cls(index)

        storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)
        resolution_node = storage.getNode("cameraResolution")
        camera_matrix = storage.getNode("cameraMatrix").mat()
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        return cls(
            index,
            resolution=(
                int(resolution_node.at(0).real()),
                int(resolution_node.at(1).real()),
            ),
            calibration=(fx, fy, cx, cy),
            vidpid=vidpid,
        )

    def _set_camera_property(self, property: int, value: int) -> None:  # noqa: A002
        """Set an opencv property to a value and assert that it changed."""
        self._camera.set(property, value)
        actual = self._camera.get(property)

        assert actual == value, (f"Failed to set property '{property}', "
                                 f"expected {value} got {actual}")

    def _set_resolution(self, resolution: Tuple[int, int]) -> None:
        """Set the camera resolution and assert that it changed."""
        width, height = resolution
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual = self._get_resolution()
        assert actual == resolution, ("Failed to set resolution expected "
                                      f"{resolution} got {actual}")

    def _get_resolution(self) -> Tuple[int, int]:
        """Get the camera resolution active in opencv."""
        return (
            int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _optimise_camera(self, vidpid: str) -> None:
        """Tweak the camera's image type and framerate to achieve the minimum frame time."""
        verified_vidpid = {'046d:0825', '046d:0807'}
        if not platform.startswith("linux"):
            # All current optimisations are for linux
            return

        # These may not improve frame time on all cameras
        if vidpid not in verified_vidpid:
            return

        camera_parameters = [
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_BUFFERSIZE, 2),
        ]

        LOGGER.debug("Optimising camera")

        for parameter, value in camera_parameters:
            try:
                self._set_camera_property(parameter, value)
            except AssertionError as e:
                LOGGER.warning(f"Failed to set property: {e}")

    def _capture_single_frame(self) -> NDArray:
        """Read a single frame from the camera's buffer."""
        ret, colour_frame = self._camera.read()
        if not ret:
            raise IOError("Failed to get frame from camera")
        return colour_frame

    def read(self, fresh: bool = True) -> NDArray:
        """
        Get another frame from the camera.

        :param fresh: Whether to flush the device's buffer before capturing
        the frame.
        """
        if fresh is True:
            for _ in range(self._buffer_length):
                # Discard a frame to remove the old frame in the buffer
                _ = self._capture_single_frame()

        return self._capture_single_frame()

    def close(self) -> None:
        """Close the underlying capture device."""
        self._camera.release()


class VideoSource(FrameSource):
    """Return frames from a video."""

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Read a video from file.

        :param filepath: The path to the video to load.
        """
        self._video = cv2.VideoCapture(str(filepath))

    def read(self, fresh: bool = True) -> NDArray:
        """
        Get the next frame from the video file.

        :param fresh: Unused.
        """
        ret, colour_frame = self._video.read()
        if not ret:
            raise IOError("Failed to get frame from video")
        return colour_frame

    def close(self) -> None:
        """Close the video file."""
        self._video.release()


class ImageSource(FrameSource):
    """Return a single image repeatedly."""

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Load an image from file.

        :param filepath: The path to the image to load.
        """
        self._frame = cv2.imread(str(filepath))

    def read(self, fresh: bool = True) -> NDArray:
        """Return the stored frame."""
        return self._frame
