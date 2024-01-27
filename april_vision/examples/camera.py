import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

from numpy.typing import NDArray

from april_vision import (CalibratedCamera, Frame, Marker, Processor,
                          USBCamera, __version__, calibrations, find_cameras,
                          generate_marker_size_mapping)
from april_vision.helpers import Base64Sender

LOGGER = logging.getLogger(__name__)


class AprilCamera:
    """
    Virtual Camera Board for detecting fiducial markers.

    Additionally, it will do pose estimation, along with some calibration
    in order to determine the spatial positon and orientation of the markers
    that it has detected.
    """

    name: str = "AprilTag Camera Board"
    use_aruco_orientation: bool = False

    @classmethod
    def discover(cls) -> Dict[str, 'AprilCamera']:
        """Discover boards that this backend can control."""
        cameras = [
            cls(camera_data.index, camera_data=camera_data,
                serial_num=camera_data.serial_num, aruco_orientation=cls.use_aruco_orientation)
            for camera_data in find_cameras(calibrations)
        ]

        return {
            camera.serial_number: camera
            for camera in cameras
        }

    def __init__(
        self,
        camera_id: int,
        camera_data: CalibratedCamera,
        serial_num: Optional[str],
        aruco_orientation: bool = False,
    ) -> None:
        """Generate a backend from the camera index and calibration data."""
        if serial_num is None:
            serial_num = f"{camera_data.name} - {camera_id}"

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
            mask_unknown_size_tags=True,
            aruco_orientation=aruco_orientation,
        )
        self._serial = serial_num
        self._name = camera_data.name

    @property
    def serial_number(self) -> str:
        """Get the serial number."""
        return self._serial

    @property
    def name(self) -> str:
        """Get the model name."""
        return self._name

    @property
    def firmware_version(self) -> Optional[str]:
        """Get the firmware version of the board."""
        return f"April camera v{__version__}"

    def make_safe(self) -> None:
        """
        Close the camera.

        The camera will no longer work after this method is called.
        """
        self._cam.close()

    # Proxy methods from USBCamera object
    def see(self, *, eager: bool = True, frame: Optional[NDArray] = None) -> List[Marker]:
        """
        Capture an image and identify fiducial markers.

        :param eager: Process the pose estimations of markers immediately,
            currently unused.
        :returns: list of markers that the camera could see.
        """
        return self._cam.see(frame=frame)

    def capture(self) -> NDArray:
        """
        Get the raw image data from the camera.

        :returns: Camera pixel data
        """
        return self._cam.capture()

    def save(self, path: Union[Path, str], *, frame: Optional[NDArray] = None) -> None:
        """Save an annotated image to a path."""
        self._cam.save(path, frame=frame)

    def set_marker_sizes(
        self,
        tag_sizes: Union[float, Dict[int, float]],
    ) -> None:
        """
        Set the size of tags that are used for pose estimation.

        If a dict is given for tag_sizes, only marker IDs that are keys of the
        dict will be detected.
        """
        self._cam.set_marker_sizes(tag_sizes)

    def set_detection_hook(self, callback: Callable[[Frame, List[Marker]], None]) -> None:
        """
        Setup a callback to be run after each dectection.
        """
        self._cam.detection_hook = callback


def setup_cameras(
    tag_sizes: Dict[Iterable[int], int],
    publish_func: Optional[Callable[[str, bytes], None]] = None,
    aruco_orientation: bool = False,
) -> Dict[str, AprilCamera]:
    """
    Find all connected cameras with calibration and configure tag sizes.

    Optionally set a callback to send a base64 encode JPEG bytestream of each
    image detection is run on.
    """
    expanded_tag_sizes = generate_marker_size_mapping(tag_sizes)

    if publish_func:
        frame_sender = Base64Sender(publish_func)

    # Set the aruco orientation flag so discovery can use it
    AprilCamera.use_aruco_orientation = aruco_orientation
    cameras = AprilCamera.discover()

    for camera in cameras.values():
        camera.set_marker_sizes(expanded_tag_sizes)
        if publish_func:
            camera.set_detection_hook(frame_sender.annotated_frame_hook)

    return cameras
