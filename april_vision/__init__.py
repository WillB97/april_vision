"""An AprilTags wrapper with camera discovery and axis conversion."""
from ._version import __version__
from .calibrations import calibrations
from .detect_cameras import find_cameras
from .frame_sources import FrameSource, USBCamera
from .marker import (CartesianCoordinates, Marker,
                     MathematicalSphericalCoordinate, Orientation,
                     PixelCoordinates, SphericalCoordinates)
from .vision import Processor

__all__ = [
    '__version__',
    'CartesianCoordinates',
    'FrameSource',
    'Marker',
    'MathematicalSphericalCoordinate',
    'Orientation',
    'PixelCoordinates',
    'Processor',
    'SphericalCoordinates',
    'USBCamera',
    'calibrations',
    'find_cameras',
]
