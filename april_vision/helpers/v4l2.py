"""
Minimal components from `v4l2py.raw` to obtain media device info.
"""

import ctypes
import fcntl
import os
from typing import Type, Union

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(dir_: int, type_: str, nr: int, size: int) -> int:
    return (
        ctypes.c_int32(dir_ << _IOC_DIRSHIFT).value
        | ctypes.c_int32(ord(type_) << _IOC_TYPESHIFT).value
        | ctypes.c_int32(nr << _IOC_NRSHIFT).value
        | ctypes.c_int32(size << _IOC_SIZESHIFT).value
    )


def _IOWR(type_: str, nr: int, size: Type[ctypes.Structure]) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, ctypes.sizeof(size))


class MediaDeviceInfo(ctypes.Structure):
    """A data structure for getting media device info"""
    _fields_ = (
        ("driver", ctypes.c_char * 16),
        ("model", ctypes.c_char * 32),
        ("serial", ctypes.c_char * 40),
        ("bus_info", ctypes.c_char * 32),
        ("media_version", ctypes.c_uint32),
        ("hw_revision", ctypes.c_uint32),
        ("driver_version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 31),
    )


MEDIA_IOC_DEVICE_INFO = _IOWR('|', 0, MediaDeviceInfo)


def read_media_device_info(path: Union[str, os.PathLike]) -> MediaDeviceInfo:
    """Given a media device path, reads its associated info"""
    info = MediaDeviceInfo()
    with open(path) as file:
        if fcntl.ioctl(file.fileno(), MEDIA_IOC_DEVICE_INFO, info):
            raise RuntimeError(f"Failed getting media device info for device: {path}")
    return info
