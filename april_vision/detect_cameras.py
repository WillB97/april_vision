"""OS-specific and generic methods for detecting attached USB cameras."""
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

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
        valid_cameras = windows_discovery(calibration_map, include_uncalibrated)
        return valid_cameras
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
            storage = cv2.FileStorage(str(calibration_file), cv2.FILE_STORAGE_READ)

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


def usable_present_devices(calibration_map: Dict[str, Path]) -> List[Tuple[str, Path, str]]:
    """Use PyUSB to detect any supported USB VID/PID combinations connected to the system."""
    import libusb_package
    try:
        usb_devices = libusb_package.find(find_all=True)
    except ValueError:
        LOGGER.warning("libusb_package failed to find a libusb backend.")
        return []

    usable_devices: List[Tuple[str, Path, str]] = []

    for dev in usb_devices:
        vidpid = f"{dev.idVendor:04x}:{dev.idProduct:04x}"
        calibration = calibration_map.get(vidpid)
        if calibration is not None:
            usable_devices.append((vidpid, calibration, calibration.stem))

    LOGGER.debug(f"Found calibration for the devices: {[dev[0] for dev in usable_devices]}")
    return usable_devices


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


def windows_discovery(
    calibration_map: Dict[str, Path],
    include_uncalibrated: bool,
) -> List[CalibratedCamera]:
    """
    Discovery method for Windows using PyUSB and the fallback discovery method.

    This cannot identify which index corresponds to the found USB VID & PID
    so is unable to provide a useful output when more than one USB camera with
    calibration is connected.
    The returned camera is a combination of the found USB VID & PID and the
    first openable camera index.
    """
    found_cameras = default_discovery()
    if include_uncalibrated:
        # We lack the information to match which camera is which so treat them
        # all as uncalibrated
        LOGGER.debug("Assuming all cameras are uncalibrated")
        return [
            CalibratedCamera(
                index=camera.index,
                name=camera.name,
                vidpid=camera.vidpid,
            ) for camera in found_cameras
        ]

    if len(found_cameras) == 0:
        return []

    selected_camera = found_cameras[0]
    LOGGER.debug(f"Selecting camera index {selected_camera.index}")

    usable_cameras = usable_present_devices(calibration_map)
    if len(usable_cameras) > 1:
        raise RuntimeError(
            "Naive calibration selection is only supported when a single compatible "
            "device is connected")
    elif len(usable_cameras) == 1:
        usable_camera = usable_cameras[0]
        return [CalibratedCamera(
            index=selected_camera.index,
            name=usable_camera[2],
            vidpid=usable_camera[0],
            calibration=usable_camera[1],
        )]
    else:
        return []


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
