from math import asin, atan2, cos, hypot, sin
from typing import List, NamedTuple, Tuple


class PixelCoordinates(NamedTuple):
    """
    Coordinates within an image made up from pixels.

    Floating point type is used to allow for subpixel detected locations
    to be represented.

    :param float x: X coordinate
    :param float y: Y coordinate
    """

    x: float
    y: float


class Coordinates(NamedTuple):
    """
    3D coordinates in space.

    :param float x: X coordinate
    :param float y: Y coordinate
    :param float z: Z coordinate
    """

    x: float
    y: float
    z: float


class Quaternion(NamedTuple):
    """A Hamiltonian quaternion."""

    w: float
    x: float
    y: float
    z: float


class Orientation(NamedTuple):
    """
    Orientation of a marker in space.

    :param yaw:   Yaw of the marker, a rotation about the vertical axis, in radians.
                  Positive values indicate a rotation clockwise from the perspective
                  of the marker.
                  Zero values have the marker facing the camera square-on.
    :param pitch: Pitch of the marker, a rotation about the transverse axis, in
                  radians.
                  Positive values indicate a rotation upwards from the perspective
                  of the marker.
                  Zero values have the marker facing the camera square-on.
    :param roll:  Roll of the marker, a rotation about the longitudinal axis,
                  in radians.
                  Positive values indicate a rotation clockwise from the perspective
                  of the marker.
                  Zero values have the marker facing the camera square-on.
    """

    yaw: float
    pitch: float
    roll: float

    @classmethod
    def from_webots_axis_angle(cls, orientation) -> 'Orientation':
        """Calculate orientation using the data from Webots' orientation data."""
        # Generate a quaternion from the axis angle
        _x, _y, _z, angle = orientation
        # Normalise the axis
        axis_mag = hypot(_x, _y, _z)
        _x, _y, _z = _x / axis_mag, _y / axis_mag, _z / axis_mag

        # Remap the axis to match the kit's coordinate system
        x, y, z = -_x, _y, -_z

        # Calculate the intrinsic Tait-Bryan angles following the z-y'-x'' convention
        # Approximately https://w.wiki/7cuk with some sign corrections,
        # adapted to axis-angle and simplified
        yaw = atan2(
            z * sin(angle) + x * y * (cos(angle) - 1),
            1 + (y ** 2 + z ** 2) * (cos(angle) - 1),
        )
        pitch = asin(x * z * (1 - cos(angle)) + y * sin(angle))
        roll = atan2(
            x * sin(angle) + y * z * (cos(angle) - 1),
            1 + (x ** 2 + y ** 2) * (cos(angle) - 1),
        )

        return cls(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

class Position(NamedTuple):
    """
    Position of a marker in space from the camera's perspective.

    :param distance:          Distance from the camera to the marker, in millimetres.
    :param horizontal_angle:  Horizontal angle from the camera to the marker, in radians.
                              Ranges from -pi to pi, with positive values indicating
                              markers to the right of the camera. Directly in front
                              of the camera is 0 rad.
    :param vertical_angle:    Vertical angle from the camera to the marker, in radians.
                              Ranges from -pi to pi, with positive values indicating
                              markers above the camera. Directly in front of the camera
                              is 0 rad.
    """

    distance: float
    horizontal_angle: float
    vertical_angle: float


PixelCorners = Tuple[PixelCoordinates, PixelCoordinates, PixelCoordinates, PixelCoordinates]


class Marker(NamedTuple):
    """
    Wrapper of a marker detection with axis and rotation calculated.

    :param id: The ID of the detected marker
    :param size: The physical size of the marker in millimeters
    :param pixel_corners: A tuple of the PixelCoordinates of the marker's corners in the frame
    :param pixel_centre: The PixelCoordinates of the marker's centre in the frame
    :param position: Position information of the marker relative to the camera
    :param orientation: Orientation information of the marker
    """

    id: int
    size: int
    pixel_corners: PixelCorners
    pixel_centre: PixelCoordinates

    position: Position = Position(0, 0, 0)
    orientation: Orientation = Orientation(0, 0, 0)

    @classmethod
    def from_webots_recognition(cls, recognition) -> 'Marker':
        """Generate a marker object using the data from Webots' Camera Recognition Object."""
        _cartesian = cls._standardise_tvec(recognition.getPosition())

        return cls(
            id=recognition.id,
            size=int(recognition.size[1] * 1000),
            pixel_corners=(  # pixel corners are not available in webots
                PixelCoordinates(0, 0),
                PixelCoordinates(0, 0),
                PixelCoordinates(0, 0),
                PixelCoordinates(0, 0),
            ),
            pixel_centre=PixelCoordinates(*recognition.getPositionOnImage()),

            position=Position(
                distance=int(hypot(*_cartesian) * 1000),
                horizontal_angle=atan2(-_cartesian.y, _cartesian.x),
                vertical_angle=atan2(_cartesian.z, _cartesian.x),
            ),

            orientation=Orientation.from_webots_axis_angle(recognition.getOrientation()),
        )

    @staticmethod
    def _standardise_tvec(tvec: List[float]) -> Coordinates:
        """
        Standardise the tvec to use the marker's coordinate system.

        The marker's coordinate system is defined as:
        - X axis is straight out of the camera
        - Y axis is to the left of the camera
        - Z axis is up
        """
        return Coordinates(tvec[0], tvec[1], tvec[2])

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self.id} distance={self.position.distance:.0f}mm "
            f"horizontal_angle={self.position.horizontal_angle:.2f}rad "
            f"vertical_angle={self.position.vertical_angle:.2f}rad size={self.size}mm>"
        )
