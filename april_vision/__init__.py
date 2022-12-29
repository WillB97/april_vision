"""An AprilTags wrapper with camera discovery and axis conversion."""
from ._version import __version__
from .detect_cameras import find_cameras
from .frame_sources import FrameSource, USBCamera
from .marker import (CartesianCoordinates, Marker, Orientation,
                     PixelCoordinates, SphericalCoordinate)
from .vision import Processor

__all__ = [
    '__version__',
    'CartesianCoordinates',
    'FrameSource',
    'Marker',
    'Orientation',
    'PixelCoordinates',
    'Processor',
    'SphericalCoordinate',
    'USBCamera',
    'find_cameras',
]
