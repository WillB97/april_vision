import re
import sys
import cv2
import json
import logging
import subprocess
from typing import Dict, List, Optional, NamedTuple
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class CameraIdentifier(NamedTuple):
    index: int  # type: ignore[assignment]
    name: str
    vidpid: str


class CalibratedCamera(NamedTuple):
    index: int  # type: ignore[assignment]
    name: str
    vidpid: str
    calibration: Optional[Path] = None


def _find_cameras(
    calibration_locations: List[str],
    include_uncalibrated: bool = False
) -> List[CalibratedCamera]:
    platform = sys.platform

    if platform.startswith("linux"):
        cameras = linux_discovery()
    elif platform.startswith("darwin"):
        cameras = mac_discovery()
    elif platform == "win32":
        cameras = windows_discovery()
    else:
        cameras = default_discovery()

    valid_cameras = match_calibrations(cameras, calibration_locations, include_uncalibrated)

    return valid_cameras


def generate_calibration_file_map(calibration_locations: List[str]) -> Dict[str, Path]:
    calibration_map: Dict[str, Path] = {}
    for location in calibration_locations:
        for calibration_file in Path(location).glob('*.xml'):
            storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)

            node = storage.getNode('pidvid')
            if node.isSeq():
                pidvids = [node.at(i).string() for i in range(node.size())]
            else:
                pidvids = [node.string()]

            for pidvid in pidvids:
                calibration_map[pidvid] = calibration_file.absolute()

    return calibration_map


def match_calibrations(
    cameras: List[CameraIdentifier],
    calibration_locations: List[str],
    include_uncalibrated: bool
) -> List[CalibratedCamera]:
    calibrated_cameras: List[CalibratedCamera] = []
    calibration_map = generate_calibration_file_map(calibration_locations)

    for camera in cameras:
        try:
            calibrated_cameras.append(CalibratedCamera(
                index=camera.index,
                name=camera.name,
                vidpid=camera.vidpid,
                calibration=calibration_map[camera.vidpid],
            ))
        except KeyError:
            if include_uncalibrated is True:
                calibrated_cameras.append(CalibratedCamera(
                    index=camera.index,
                    name=camera.name,
                    vidpid=camera.vidpid,
                ))

    return calibrated_cameras


def linux_discovery() -> List[CameraIdentifier]:
    # Get a list of all potential cameras
    potential_cameras = []
    for index in range(10):
        if Path(f'/dev/video{index}').exists():
            potential_cameras.append(index)

    # Filter the potential cameras to ones that work
    valid_cameras = []
    for index in potential_cameras:
        capture = cv2.VideoCapture(f'/dev/video{index}')
        if capture.isOpened():
            valid_cameras.append(index)
        capture.release()

    # Match pid:vid to the path
    cameras = []
    for index in valid_cameras:
        name = Path(
                f'/sys/class/video4linux/video{index}/name'
            ).read_text().strip()

        uevent_file = Path(f'/sys/class/video4linux/video{index}/device/uevent').read_text()
        m = re.search(r'PRODUCT=([0-9a-f]{1,4})\/([0-9a-f]{1,4})', uevent_file)

        if m is None:
            continue

        vid = int(m.groups()[0], 16)
        pid = int(m.groups()[1], 16)

        vidpid = f'{vid:04x}:{pid:04x}'

        LOGGER.debug(f"Found camera at index {index}: {name}")
        cameras.append(CameraIdentifier(
            index=index,
            name=name,
            vidpid=vidpid,
        ))

    return cameras


def mac_discovery() -> List[CameraIdentifier]:
    camera_info = json.loads(
        subprocess.check_output(['system_profiler', '-json', 'SPCameraDataType']),
    )
    camera_list = camera_info["SPCameraDataType"]
    cameras = []

    for index, camera in enumerate(camera_list):
        try:
            name = camera["_name"]
            camera_data = camera['spcamera_model-id']

            m = re.search(r'VendorID_([0-9]{1,5}) ProductID_([0-9]{1,5})', camera_data)

            if m is None:
                # Facetime cameras have no PID or VID
                vidpid = camera_data
            else:
                vid = int(m.groups()[0], 10)
                pid = int(m.groups()[1], 10)

                vidpid = f'{vid:04x}:{pid:04x}'

            LOGGER.debug(f"Found camera at index {index}: {name}")
            cameras.append(CameraIdentifier(index=index, name=name, vidpid=vidpid))
        except KeyError:
            LOGGER.warning(f"Camera {index} had missing fields: {camera}")

    return cameras


def windows_discovery() -> List[CameraIdentifier]:
    return default_discovery()


def default_discovery() -> List[CameraIdentifier]:
    found_cameras = []
    for camera_id in range(8):
        capture = cv2.VideoCapture(camera_id)
        if capture.isOpened():
            found_cameras.append(CameraIdentifier(
                index=camera_id,
                name=str(camera_id),
                vidpid='',
            ))
        capture.release()

    return found_cameras
