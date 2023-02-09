import typing
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

Mat = np.ndarray[int, np.dtype]

FILE_STORAGE_APPEND: int
FILE_STORAGE_BASE64: int
FILE_STORAGE_FORMAT_AUTO: int
FILE_STORAGE_FORMAT_JSON: int
FILE_STORAGE_FORMAT_MASK: int
FILE_STORAGE_FORMAT_XML: int
FILE_STORAGE_FORMAT_YAML: int
FILE_STORAGE_INSIDE_MAP: int
FILE_STORAGE_MEMORY: int
FILE_STORAGE_NAME_EXPECTED: int
FILE_STORAGE_READ: int
FILE_STORAGE_UNDEFINED: int
FILE_STORAGE_VALUE_EXPECTED: int
FILE_STORAGE_WRITE: int
FILE_STORAGE_WRITE_BASE64: int

FONT_HERSHEY_COMPLEX: int
FONT_HERSHEY_COMPLEX_SMALL: int
FONT_HERSHEY_DUPLEX: int
FONT_HERSHEY_PLAIN: int
FONT_HERSHEY_SCRIPT_COMPLEX: int
FONT_HERSHEY_SCRIPT_SIMPLEX: int
FONT_HERSHEY_SIMPLEX: int
FONT_HERSHEY_TRIPLEX: int
FONT_ITALIC: int

CAP_PROP_APERTURE: int
CAP_PROP_ARAVIS_AUTOTRIGGER: int
CAP_PROP_AUTOFOCUS: int
CAP_PROP_AUTO_EXPOSURE: int
CAP_PROP_AUTO_WB: int
CAP_PROP_BACKEND: int
CAP_PROP_BACKLIGHT: int
CAP_PROP_BITRATE: int
CAP_PROP_BRIGHTNESS: int
CAP_PROP_BUFFERSIZE: int
CAP_PROP_CHANNEL: int
CAP_PROP_CODEC_PIXEL_FORMAT: int
CAP_PROP_CONTRAST: int
CAP_PROP_CONVERT_RGB: int
CAP_PROP_DC1394_MAX: int
CAP_PROP_DC1394_MODE_AUTO: int
CAP_PROP_DC1394_MODE_MANUAL: int
CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO: int
CAP_PROP_DC1394_OFF: int
CAP_PROP_EXPOSURE: int
CAP_PROP_EXPOSUREPROGRAM: int
CAP_PROP_FOCUS: int
CAP_PROP_FORMAT: int
CAP_PROP_FOURCC: int
CAP_PROP_FPS: int
CAP_PROP_FRAME_COUNT: int
CAP_PROP_FRAME_HEIGHT: int
CAP_PROP_FRAME_WIDTH: int
CAP_PROP_GAIN: int
CAP_PROP_GAMMA: int
CAP_PROP_GIGA_FRAME_HEIGH_MAX: int
CAP_PROP_GIGA_FRAME_OFFSET_X: int
CAP_PROP_GIGA_FRAME_OFFSET_Y: int
CAP_PROP_GIGA_FRAME_SENS_HEIGH: int
CAP_PROP_GIGA_FRAME_SENS_WIDTH: int
CAP_PROP_GIGA_FRAME_WIDTH_MAX: int
CAP_PROP_GPHOTO2_COLLECT_MSGS: int
CAP_PROP_GPHOTO2_FLUSH_MSGS: int
CAP_PROP_GPHOTO2_PREVIEW: int
CAP_PROP_GPHOTO2_RELOAD_CONFIG: int
CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE: int
CAP_PROP_GPHOTO2_WIDGET_ENUMERATE: int
CAP_PROP_GSTREAMER_QUEUE_LENGTH: int
CAP_PROP_GUID: int
CAP_PROP_HUE: int
CAP_PROP_IMAGES_BASE: int
CAP_PROP_IMAGES_LAST: int
CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD: int
CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ: int
CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT: int
CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE: int
CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE: int
CAP_PROP_INTELPERC_PROFILE_COUNT: int
CAP_PROP_INTELPERC_PROFILE_IDX: int
CAP_PROP_IOS_DEVICE_EXPOSURE: int
CAP_PROP_IOS_DEVICE_FLASH: int
CAP_PROP_IOS_DEVICE_FOCUS: int
CAP_PROP_IOS_DEVICE_TORCH: int
CAP_PROP_IOS_DEVICE_WHITEBALANCE: int
CAP_PROP_IRIS: int
CAP_PROP_ISO_SPEED: int
CAP_PROP_MODE: int
CAP_PROP_MONOCHROME: int
CAP_PROP_OPENNI2_MIRROR: int
CAP_PROP_OPENNI2_SYNC: int
CAP_PROP_OPENNI_APPROX_FRAME_SYNC: int
CAP_PROP_OPENNI_BASELINE: int
CAP_PROP_OPENNI_CIRCLE_BUFFER: int
CAP_PROP_OPENNI_FOCAL_LENGTH: int
CAP_PROP_OPENNI_FRAME_MAX_DEPTH: int
CAP_PROP_OPENNI_GENERATOR_PRESENT: int
CAP_PROP_OPENNI_MAX_BUFFER_SIZE: int
CAP_PROP_OPENNI_MAX_TIME_DURATION: int
CAP_PROP_OPENNI_OUTPUT_MODE: int
CAP_PROP_OPENNI_REGISTRATION: int
CAP_PROP_OPENNI_REGISTRATION_ON: int
CAP_PROP_PAN: int
CAP_PROP_POS_AVI_RATIO: int
CAP_PROP_POS_FRAMES: int
CAP_PROP_POS_MSEC: int
CAP_PROP_PVAPI_BINNINGX: int
CAP_PROP_PVAPI_BINNINGY: int
CAP_PROP_PVAPI_DECIMATIONHORIZONTAL: int
CAP_PROP_PVAPI_DECIMATIONVERTICAL: int
CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE: int
CAP_PROP_PVAPI_MULTICASTIP: int
CAP_PROP_PVAPI_PIXELFORMAT: int
CAP_PROP_RECTIFICATION: int
CAP_PROP_ROLL: int
CAP_PROP_SAR_DEN: int
CAP_PROP_SAR_NUM: int
CAP_PROP_SATURATION: int
CAP_PROP_SETTINGS: int
CAP_PROP_SHARPNESS: int
CAP_PROP_SPEED: int
CAP_PROP_TEMPERATURE: int
CAP_PROP_TILT: int
CAP_PROP_TRIGGER: int
CAP_PROP_TRIGGER_DELAY: int
CAP_PROP_VIEWFINDER: int
CAP_PROP_WB_TEMPERATURE: int
CAP_PROP_WHITE_BALANCE_BLUE_U: int
CAP_PROP_WHITE_BALANCE_RED_V: int
CAP_PROP_ZOOM: int

COLOR_BGR2GRAY: int

IMREAD_ANYCOLOR: int
IMREAD_ANYDEPTH: int
IMREAD_COLOR: int
IMREAD_GRAYSCALE: int
IMREAD_IGNORE_ORIENTATION: int
IMREAD_LOAD_GDAL: int
IMREAD_REDUCED_COLOR_2: int
IMREAD_REDUCED_COLOR_4: int
IMREAD_REDUCED_COLOR_8: int
IMREAD_REDUCED_GRAYSCALE_2: int
IMREAD_REDUCED_GRAYSCALE_4: int
IMREAD_REDUCED_GRAYSCALE_8: int
IMREAD_UNCHANGED: int

FILE_NODE_NONE: int
FILE_NODE_INT: int
FILE_NODE_REAL: int
FILE_NODE_FLOAT: int
FILE_NODE_STR: int
FILE_NODE_STRING: int
FILE_NODE_SEQ: int
FILE_NODE_MAP: int
FILE_NODE_TYPE_MASK: int
FILE_NODE_FLOW: int
FILE_NODE_UNIFORM: int
FILE_NODE_EMPTY: int
FILE_NODE_NAMED: int


def cvtColor(src: Mat, code: int, dts: Mat = ..., dstCn: int = ...) -> Mat: ...
def imread(filename: str, flags: int = ...) -> Mat: ...
def imwrite(filename: str, img: Mat, params: typing.List[int] = ...) -> bool: ...
def imencode(ext: str, img: Mat, params: typing.List[int] = ...) -> typing.Tuple[bool, Mat]: ...
def polylines(img: Mat, pts: typing.List[NDArray], isClosed: bool, color: typing.Tuple[int, int, int], thickness: int = ..., lineType: int = ..., shift: int = ...) -> Mat: ...
def putText(img: Mat, text: str, org: NDArray, fontFace: int, fontScale: float, color: typing.Tuple[int, int, int], thickness: int = ..., lineType: int = ..., bottomLeftOrigin: bool = ...) -> Mat: ...
def VideoWriter_fourcc(c1: str, c2: str, c3: str, c4: str) -> int: ...
def imshow(winname: str, mat: Mat) -> None: ...
def waitKey(delay: int = 0) -> int: ...
def destroyAllWindows() -> None: ...
def flip(src: Mat, flipCode: int) -> Mat: ...
def resize(src: Mat, dsize: Tuple[int, int]) -> Mat: ...


class Node:
    def __init__(self, payload: Any) -> None: ...
    def isSeq(self) -> bool: ...
    def isString(self) -> bool: ...
    def at(self, i0: int) -> 'Node': ...
    def size(self) -> int: ...
    def string(self) -> str: ...
    def mat(self) -> Mat: ...
    def real(self) -> float: ...


class FileStorage:
    def __init__(self, filename: str, flags: int, encoding: str = ...) -> None: ...
    def getNode(self, nodename: str) -> Node: ...
    def write(self, nodename: str, payload: typing.Any) -> None: ...
    def release(self) -> None: ...
    def startWriteStruct(self, nodename: str, flags: int) -> None: ...
    def endWriteStruct(self) -> None: ...


class VideoCapture:
    def __init__(self, index: typing.Union[str, int], apiPreference: int = ...) -> None: ...
    def set(self, propId: int, value: float) -> None: ...
    def get(self, propId: int) -> float: ...
    def release(self) -> None: ...
    def isOpened(self) -> bool: ...
    def read(self) -> typing.Tuple[bool, Mat]: ...


class VideoWriter:
    def __init__(self, filename: str, fourcc: int, fps: float, frameSize: Tuple[int, int]) -> None: ...
    def write(self, frame: Mat) -> None: ...
    def release(self) -> None: ...
