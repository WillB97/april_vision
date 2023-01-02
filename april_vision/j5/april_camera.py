"""j5 integration for april_vision."""
import base64
import logging
import os
from pathlib import Path
from threading import Thread
from typing import Callable, Dict, Iterable, List, Optional, Set, Type, Union

import cv2
import numpy as np
from j5.backends import Backend
from j5.boards import Board
from j5.components.component import Component
from numpy.typing import NDArray

from .._version import __version__
from ..detect_cameras import CalibratedCamera, find_cameras
from ..frame_sources import USBCamera
from ..marker import Marker, MarkerType
from ..vision import Frame, Processor

LOGGER = logging.getLogger(__name__)


class AprilCameraBoard(Board):
    """
    Virtual Camera Board for detecting fiducial markers.

    Additionally, it will do pose estimation, along with some calibration
    in order to determine the spatial positon and orientation of the markers
    that it has detected.
    """

    name: str = "AprilTag Camera Board"
    _backend: 'AprilTagHardwareBackend'

    def __init__(self, serial: str, backend: 'AprilTagHardwareBackend'):
        self._serial = serial
        self._backend = backend

    @property
    def serial_number(self) -> str:
        """Get the serial number."""
        return self._serial

    @property
    def firmware_version(self) -> Optional[str]:
        """Get the firmware version of the board."""
        return self._backend.firmware_version

    def make_safe(self) -> None:
        """
        Close the camera.

        The camera will no longer work after this method is called.
        """
        self._backend.close_camera()

    @staticmethod
    def supported_components() -> Set[Type[Component]]:
        """List the types of components supported by this board."""
        return set()

    # Proxy methods from MarkerCamera object
    def see(self, *, eager: bool = True, frame: Optional[NDArray] = None) -> List[Marker]:
        """
        Capture an image and identify fiducial markers.

        :param eager: Process the pose estimations of markers immediately,
            currently unused.
        :returns: list of markers that the camera could see.
        """
        return self._backend.see(frame=frame)

    def see_ids(self, *, frame: Optional[NDArray] = None) -> List[int]:
        """
        Capture an image and identify fiducial markers.

        This method returns just the marker IDs that are visible.
        :returns: A list of IDs for the markers that were visible.
        """
        return self._backend.see_ids(frame=frame)

    def capture(self) -> NDArray:
        """
        Get the raw image data from the camera.

        :returns: Camera pixel data
        """
        return self._backend.capture_frame()

    def save(self, path: Union[Path, str], *, frame: Optional[NDArray] = None) -> None:
        """Save an annotated image to a path."""
        self._backend.save_annotated_image(path, frame=frame)


class AprilTagHardwareBackend(Backend):
    """An April Vision Hardware backend."""

    board = AprilCameraBoard
    marker_type = MarkerType.APRILTAG_36H11

    @classmethod
    def discover(cls) -> Set[Board]:
        """Discover boards that this backend can control."""
        return {
            AprilCameraBoard(
                f"{camera_data.name} - {camera_data.index}",
                cls(camera_data.index, camera_data=camera_data),
            )
            for camera_data in find_cameras(
                os.environ.get('OPENCV_CALIBRATIONS', '.').split(':'))
        }

    def __init__(self, camera_id: int, camera_data: CalibratedCamera) -> None:
        """Generate a backend from the camera index and calibration data."""
        camera_source = USBCamera.from_calibration_file(
            camera_id,
            calibration_file=camera_data.calibration,
            vidpid=camera_data.vidpid,
        )
        self._cam = Processor(
            camera_source,
            calibration=camera_source.calibration,
            name=camera_data.name,
            vidpid=camera_data.vidpid,
        )
        self._marker_offset = 0
        self._cam.marker_filter = self.marker_filter
        self._mqtt_publish: Optional[Callable[[str, Union[bytes, str]], None]] = None
        self._cam.detection_hook = self.annotated_frame_hook

    def see(
        self,
        *,
        frame: Optional[NDArray] = None,
    ) -> List[Marker]:
        """Get markers that the camera can see."""
        return self._cam.see(frame=frame)

    def save_annotated_image(
        self, file: Union[Path, str], *, frame: Optional[NDArray] = None,
    ) -> None:
        """Save an annotated image to a file."""
        self._cam.save(file, frame=frame)

    def see_ids(self, *, frame: Optional[NDArray] = None) -> List[int]:
        """
        Get a list of visible marker IDs.

        :returns: List of marker IDs that were visible.
        """
        return self._cam.see_ids(frame=frame)

    def capture_frame(self) -> NDArray:
        """
        Get the raw image data from the camera.

        :returns: Camera pixel data
        """
        return self._cam.capture()

    @property
    def firmware_version(self) -> Optional[str]:
        """The firmware version of the board."""
        return f"April camera v{__version__}"

    def close_camera(self) -> None:
        """Close the camera object."""
        self._cam.close()

    def set_marker_sizes(
        self,
        marker_sizes: Dict[Iterable[int], int],
        marker_offset: int = 0,
    ) -> None:
        """Set the sizes of all the markers used in the game."""
        # store marker offset to be used by the filter
        self._marker_offset = marker_offset
        # Reset previously stored sizes
        self._cam.tag_sizes = {}
        for marker_ids, marker_size in marker_sizes.items():
            # Unroll generators to give direct lookup
            for marker_id in marker_ids:
                # Convert to meters
                self._cam.tag_sizes[marker_id + marker_offset] = float(marker_size) / 1000

    def marker_filter(self, markers: List[Marker]) -> List[Marker]:
        """Apply marker offset and remove markers that are not in the game."""
        filtered_markers: List[Marker] = []
        if not isinstance(self._cam.tag_sizes, dict):
            for marker in markers:
                marker._id -= self._marker_offset
                filtered_markers.append(marker)
            return filtered_markers

        for marker in markers:
            if marker._id in self._cam.tag_sizes.keys():
                marker._id -= self._marker_offset
                filtered_markers.append(marker)

        return filtered_markers

    def annotated_frame_hook(self, frame: Frame, markers: List[Marker]) -> None:
        """
        During _detect annotate a copy of the frame and send it over MQTT.

        The image data is a base64 JPEG.
        """
        if self._mqtt_publish is None:
            return

        copied_frame = Frame(
            np.array(frame.grey_frame, copy=True),
            np.array(frame.colour_frame, copy=True),
        )
        annotated_frame = self._cam._annotate(copied_frame, markers)

        thread = Thread(
            name="Image send",
            target=self.background_encode_and_send,
            args=(annotated_frame.colour_frame,),
            daemon=True,
        )
        thread.start()

    def background_encode_and_send(self,  frame: NDArray[np.uint8]) -> None:
        """
        Handle converting a frame to a base64 bytestring and sending over MQTT.

        To be run as a thread target.
        """
        if self._mqtt_publish is None:
            return

        encoded_frame = self.base64_encode_frame(frame)
        if encoded_frame is None:
            return

        encoded_frame = b'data:image/jpeg;base64, ' + encoded_frame

        # The message is sent in a background thread
        try:
            self._mqtt_publish(
                'camera/annotated',
                encoded_frame,
            )
        except ValueError:
            pass

    def base64_encode_frame(self, frame: NDArray[np.uint8]) -> Optional[bytes]:
        """Convert image frame to base64 bytestring."""
        ret, image = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        image_bytes = image.tobytes()
        return base64.b64encode(image_bytes)
