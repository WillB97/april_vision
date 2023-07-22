"""OS-specific and generic methods for detecting attached USB cameras."""
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import cv2

LOGGER = logging.getLogger(__name__)


class CameraIdentifier(NamedTuple):
    """A tuple to store information of a discovered camera."""

    index: int  # type: ignore[assignment]
    name: str
    vidpid: str


class CalibratedCamera(NamedTuple):
    """A tuple to store camera and calibration information of a discovered camera."""

    index: int  # type: ignore[assignment]
    name: str
    vidpid: str
    calibration: Optional[Path] = None


def find_cameras(
    calibration_locations: List[str],
    include_uncalibrated: bool = False,
) -> List[CalibratedCamera]:
    """Find the available cameras using OS-specific discovery where available."""
    platform = sys.platform

    calibration_map = generate_calibration_file_map(calibration_locations)

    if platform.startswith("linux"):
        cameras = linux_discovery()
    elif platform.startswith("darwin"):
        cameras = mac_discovery()
    elif platform == "win32":
        cameras = windows_discovery()
    else:
        cameras = default_discovery()

    valid_cameras = match_calibrations(cameras, calibration_map, include_uncalibrated)

    return valid_cameras


def generate_calibration_file_map(calibration_locations: List[str]) -> Dict[str, Path]:
    """
    Generate map of USB VID & PID to calibration file.

    Uses vidpid tag in XML files.
    """
    calibration_map: Dict[str, Path] = {}
    # iterate calibration locations backword so that so that earlier locations
    # in the list get higher precedence
    for location in reversed(calibration_locations):
        for calibration_file in Path(location).glob('*.xml'):
            try:
                storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)
            except SystemError:
                LOGGER.debug(f"Unable to read potential calibration file: {calibration_file}")
                continue

            node = storage.getNode('vidpid')
            if node.isSeq():
                pidvids = [node.at(i).string() for i in range(node.size())]
            elif node.isString():
                pidvids = [node.string()]
            else:
                # This file lacks any vidpids
                continue

            for pidvid in pidvids:
                calibration_map[pidvid] = calibration_file.absolute()
    LOGGER.debug(f"Calibrations found for: {list(calibration_map.keys())}")

    return calibration_map


def match_calibrations(
    cameras: List[CameraIdentifier],
    calibration_map: Dict[str, Path],
    include_uncalibrated: bool,
) -> List[CalibratedCamera]:
    """Filter found cameras to those that matching calibration files' USB VID & PID."""
    calibrated_cameras: List[CalibratedCamera] = []
    uncalibrated_cameras: List[CalibratedCamera] = []

    for camera in cameras:
        try:
            calibrated_cameras.append(CalibratedCamera(
                index=camera.index,
                name=camera.name,
                vidpid=camera.vidpid,
                calibration=calibration_map[camera.vidpid],
            ))
            LOGGER.debug(
                f"Found calibration for {camera.name} in {calibration_map[camera.vidpid]}")
        except KeyError:
            LOGGER.debug(f"No calibration found for for {camera.name}")
            if include_uncalibrated is True:
                uncalibrated_cameras.append(CalibratedCamera(
                    index=camera.index,
                    name=camera.name,
                    vidpid=camera.vidpid,
                ))

    return calibrated_cameras + uncalibrated_cameras


def linux_discovery() -> List[CameraIdentifier]:
    """
    Discovery method for Linux using Video4Linux.

    This matches camera indexes to their USB VID & PID and omits indexes
    that are not the actual camera.
    """
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
                f'/sys/class/video4linux/video{index}/name',
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
    """
    Discovery method for MacOS using system_profiler.

    This matches camera indexes to their USB VID & PID as long as cameras are
    not attached after cv2 has been imported.
    """
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
    """
    Discovery method for Windows using windowsRT API.

    Results are only valid for the MSMF opencv backend.
    This matches camera indexes to their USB VID & PID and omits indexes
    that are not USB cameras.
    """
    import asyncio

    import winsdk.windows.devices.enumeration as windows_devices  # type: ignore

    async def get_camera_info():  # type: ignore
        device_class = windows_devices.DeviceClass.VIDEO_CAPTURE
        return await windows_devices.DeviceInformation.find_all_async(device_class)

    connected_cameras = asyncio.run(get_camera_info())  # type: ignore

    cameras = []
    for index, device in enumerate(connected_cameras):
        m = re.search(r'USB#VID_([0-9,A-F]{4})&PID_([0-9,A-F]{4})', device.id)
        if m is None:
            continue

        vid = int(m.groups()[0], 16)
        pid = int(m.groups()[1], 16)

        vidpid = f'{vid:04x}:{pid:04x}'

        LOGGER.debug(f"Found camera at index {index}: {device.name}")

        cameras.append(CameraIdentifier(
            index=index,
            name=device.name,
            vidpid=vidpid,
        ))

    return cameras


def default_discovery() -> List[CameraIdentifier]:
    """
    A fallback discovery method for when there is not an OS specific one available.

    This cannot identify the USB VID & PID of the camera and only provides
    information on the openable indexes.
    """
    found_cameras = []
    for camera_id in range(8):
        capture = cv2.VideoCapture(camera_id)
        if capture.isOpened():
            LOGGER.debug(f"Found camera at index {camera_id}")
            found_cameras.append(CameraIdentifier(
                index=camera_id,
                name=str(camera_id),
                vidpid='',
            ))
        capture.release()

    return found_cameras
