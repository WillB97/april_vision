import re
import sys
import cv2
import json
import subprocess
from typing import List, Optional, NamedTuple
from pathlib import Path


class CameraIdentifier(NamedTuple):
    index: int
    name: str
    pidvid: str


class CalibratedCamera(NamedTuple):
    index: int
    name: str
    pidvid: str
    calibration: Optional[Path] = None


def _find_cameras(calibration_locations: List[str], include_uncalibrated: bool = False) -> List[CalibratedCamera]:
    platform = sys.platform()

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


def match_calibrations(cameras: List[CameraIdentifier], calibration_locations: List[str], include_uncalibrated: bool):
    pass


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

        pid = int(m.groups()[0], 16)
        vid = int(m.groups()[1], 16)

        pidvid = f'{pid:04x}:{vid:04x}'

        cameras.append(CameraIdentifier(
            index=index,
            name=name,
            pidvid=pidvid,
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
            cameras.append(CameraIdentifier(index=index, name=camera["_name"]))
        except KeyError:
            pass  # TODO log that we got a camera with no name

    return cameras


def windows_discovery() -> List[CameraIdentifier]:
    pass


def default_discovery() -> List[CameraIdentifier]:
    pass


# def get_pidvid_cal_map(dir):
#     mapping = []

#     for file in dir:
#         s = april_vision.Camera.from_calibration_file(0, filename)

#         n = s.getNode('pidvid')
#         if n.isSeq():
#             pidvids = [n.at(i).string() for i in range(n.size())]
#         else:
#             pidvids = [n.string()]

#         for pidvid in pidvids:
#             mapping.append(
#                 (pidvid, filename)
#             )
