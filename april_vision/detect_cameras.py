"""OS-specific and generic methods for detecting attached USB cameras."""
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import cv2

LOGGER = logging.getLogger(__name__)


class CameraIdentifier(NamedTuple):
    """A tuple to store information of a discovered camera."""

    index: int  # type: ignore[assignment]
    name: str
    vidpid: str
    serial_num: Optional[str] = None


class CalibratedCamera(NamedTuple):
    """A tuple to store camera and calibration information of a discovered camera."""

    index: int  # type: ignore[assignment]
    name: str
    vidpid: str
    serial_num: Optional[str] = None
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
                serial_num=camera.serial_num,
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
                    serial_num=camera.serial_num,
                ))

    return calibrated_cameras + uncalibrated_cameras


def linux_discovery() -> List[CameraIdentifier]:
    """
    Discovery method for Linux using Video4Linux.

    This matches camera indexes to their USB VID & PID and omits indexes
    that are not the actual camera.
    """
    # Filter the potential cameras to ones that work
    valid_cameras = []
    for dev in Path('/dev').glob('video*'):
        try:
            index = int(dev.stem.replace('video', ''))
        except ValueError:
            # Not a video device
            continue

        # Check if the device is a valid camera
        capture = cv2.VideoCapture(str(dev))
        if capture.isOpened():
            valid_cameras.append(index)
        capture.release()

    # Match pid:vid to the path
    cameras = []
    for index in valid_cameras:
        cam_path = Path(f'/sys/class/video4linux/video{index}')
        name = (cam_path / 'name').read_text().strip()

        uevent_file = (cam_path / 'device/uevent').read_text()
        m = re.search(r'PRODUCT=([0-9a-f]{1,4})\/([0-9a-f]{1,4})', uevent_file)

        if m is None:
            continue

        vid = int(m.groups()[0], 16)
        pid = int(m.groups()[1], 16)

        vidpid = f'{vid:04x}:{pid:04x}'

        serial_num = None
        # This path is not what it first appears, due ot symbolic links.
        # The up traversal follows the symbolic link across into the USB tree
        # at the location of this device, which allows us to inspect the USB
        # descriptor for the serial number.
        serial_file = cam_path / 'device/../serial'
        if serial_file.exists():
            serial_num = serial_file.read_text().strip()

        LOGGER.debug(f"Found camera at index {index}: {name} (serial: {serial_num})")
        cameras.append(CameraIdentifier(
            index=index,
            name=name,
            vidpid=vidpid,
            serial_num=serial_num,
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
    # Preserve devices ordering on the system
    # see AVCaptureDevice::uniqueID property documentation for more info
    # From https://github.com/opencv/opencv/blob/4.11.0/modules/videoio/src/cap_avfoundation_mac.mm#L377
    camera_list.sort(key=lambda x: x["spcamera_unique-id"])
    cameras = []

    for index, camera in enumerate(camera_list):
        try:
            name = camera["_name"]
            camera_data = camera['spcamera_model-id']
            # Use caution, this identifier follows the port not the camera
            unique_id = camera['spcamera_unique-id']

            m = re.search(r'VendorID_([0-9]{1,5}) ProductID_([0-9]{1,5})', camera_data)

            if m is None:
                # Facetime cameras have no PID or VID
                vidpid = camera_data
            else:
                vid = int(m.groups()[0], 10)
                pid = int(m.groups()[1], 10)

                vidpid = f'{vid:04x}:{pid:04x}'

            LOGGER.debug(f"Found camera at index {index}: {name}")
            cameras.append(CameraIdentifier(
                index=index,
                name=name,
                vidpid=vidpid,
                serial_num=unique_id,
            ))
        except KeyError:
            LOGGER.warning(f"Camera {index} had missing fields: {camera}")

    # Attempt to get the USB serial numbers for the cameras
    return mac_insert_usb_serials(cameras)


def mac_insert_usb_serials(cameras: List[CameraIdentifier]) -> List[CameraIdentifier]:
    """
    Insert the USB serial numbers into the camera list.

    system_profiler reports only the USB path and vidpid.
    We use the USB path to get the serial number reported in the USB descriptor.
    """

    class USBDevice(NamedTuple):
        """A tuple to store information of a USB device."""

        location_id: str
        vendor_id: str
        product_id: str
        serial_num: str

    def process_usb_hub(usb_hub: List[Any]) -> Dict[str, USBDevice]:
        """Recursively process USB hubs to find serial numbers."""
        usb_serials = {}
        for device in usb_hub:
            if '_items' in device.keys():
                usb_serials.update(process_usb_hub(device['_items']))
            try:
                serial_num = device['serial_num']
                usb_path = device['location_id'].split(' ')[0][2:].lstrip('0')
                usb_vid = device['vendor_id'].split(' ')[0][2:]
                usb_pid = device['product_id'][2:]
            except KeyError:
                continue

            usb_serials[usb_path] = USBDevice(
                location_id=usb_path,
                vendor_id=usb_vid,
                product_id=usb_pid,
                serial_num=serial_num,
            )
            LOGGER.debug(f"Found USB device: {usb_vid}:{usb_pid} {serial_num} 0x{usb_path}")

        return usb_serials

    # Grab USB tree and process it into a dictionary
    usb_tree = subprocess.check_output(['system_profiler', '-json', 'SPUSBDataType'])
    try:
        usb_data = json.loads(usb_tree)['SPUSBDataType']
    except (json.JSONDecodeError, KeyError):
        LOGGER.warning("Unable to decode USB tree")
        return cameras

    # Create a dictionary of USB paths to serial numbers
    usb_serials = process_usb_hub(usb_data)

    # iterate over cameras updating the serial numbers
    for index, camera in enumerate(cameras):
        # skip non-USB cameras
        if not camera.serial_num or not camera.serial_num.startswith('0x'):
            continue

        # split unique id to extract just usb path
        usb_path = camera.serial_num[2:-8]
        # strip any leading zeros
        usb_path = usb_path.lstrip('0')

        device_info = usb_serials.get(usb_path)
        if device_info is None:
            LOGGER.warning(f"Unable to find USB device for camera {camera.name}")
            continue

        # Check this path matches the device we are expecting
        vidpid = f'{device_info.vendor_id}:{device_info.product_id}'
        if camera.vidpid != vidpid:
            LOGGER.warning(
                f"Camera {camera.name} has mismatched USB path: {vidpid} != {camera.vidpid}"
            )
            continue

        # Update the camera with the serial number
        cameras[index] = camera._replace(serial_num=device_info.serial_num)

    return cameras


def windows_discovery() -> List[CameraIdentifier]:
    """
    Discovery method for Windows using windowsRT API.

    Results are only valid for the MSMF opencv backend.
    This matches camera indexes to their USB VID & PID and omits indexes
    that are not USB cameras.
    """
    import asyncio

    assert sys.platform == 'win32', "This method is only for Windows"

    import winrt.windows.devices.enumeration as windows_devices  # type: ignore[import-not-found,unused-ignore]
    from winrt.windows.foundation import (  # type: ignore[import-not-found,unused-ignore]
        IPropertyValue,
    )

    async def get_parent_id(device_container):  # type: ignore
        device_id = device_container.properties['System.Devices.DeviceInstanceId']
        device_id = IPropertyValue._from(device_id).get_string()

        device = await windows_devices.DeviceInformation.create_from_id_async(
            device_id,
            ['System.Devices.Parent'],
            windows_devices.DeviceInformationKind.DEVICE,
        )

        assert device is not None, "Unable to get parent device ID"
        assert device.properties is not None, "Unable to get parent device properties"
        return IPropertyValue._from(device.properties['System.Devices.Parent']).get_string()

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

        # Get USB serial number from the parent device ID
        parent_id = asyncio.run(get_parent_id(device))  # type: ignore
        serial_num = None
        parent_id_parts = parent_id.split('\\')
        if len(parent_id_parts) == 3 and parent_id_parts[0] == 'USB':
            serial_num = parent_id_parts[2]

        cameras.append(CameraIdentifier(
            index=index,
            name=device.name,
            vidpid=vidpid,
            serial_num=serial_num,
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
