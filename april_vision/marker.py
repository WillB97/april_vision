"""
Classes for marker detections and various axis representations.

Setting the environment variable ZOLOTO_LEGACY_AXIS uses the axis that were
used in zoloto<0.9.0. Otherwise the conventional right-handed axis is used
where x is forward, y is left and z is upward.
"""
import os
from enum import Enum
from math import acos, atan2, pi
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
from numpy.linalg import norm as hypotenuse
from numpy.typing import NDArray
from pyapriltags import Detection
from pyquaternion import Quaternion


class MarkerType(Enum):
    """
    The available tag families.

    To support Apriltag 2 libraries use tag36h11.
    """

    APRILTAG_16H5 = 'tag16h5'
    APRILTAG_25H9 = 'tag25h9'
    APRILTAG_36H11 = 'tag36h11'
    APRILTAG_21H7 = 'tagCircle21h7'
    APRILTAG_49H12 = 'tagCircle49h12'
    APRILTAG_48H12 = 'tagCustom48h12'
    APRILTAG_41H12 = 'tagStandard41h12'
    APRILTAG_52H13 = 'tagStandard52h13'


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


class CartesianCoordinates(NamedTuple):
    """
    A 3 dimesional cartesian coordinate in the standard right-handed cartesian system.

    Origin is at the camera.

    The X axis extends directly away from the camera. Zero is at the camera.
    Increasing values indicate greater distance from the camera.

    The Y axis is horizontal relative to the camera's perspective, i.e: right
    to left within the frame of the image. Zero is at the centre of the image.
    Increasing values indicate greater distance to the left.

    The Z axis is vertical relative to the camera's perspective, i.e: down to
    up within the frame of the image. Zero is at the centre of the image.
    Increasing values indicate greater distance above the centre of the image.

    More information: https://w.wiki/5zbE

    Legacy:
    The X axis is horizontal relative to the camera's perspective, i.e: left &
    right within the frame of the image. Zero is at the centre of the image.
    Increasing values indicate greater distance to the right.

    The Y axis is vertical relative to the camera's perspective, i.e: up & down
    within the frame of the image. Zero is at the centre of the image.
    Increasing values indicate greater distance below the centre of the image.

    The Z axis extends directly away from the camera. Zero is at the camera.
    Increasing values indicate greater distance from the camera.

    These match traditional cartesian coordinates when the camera is facing
    upwards.


    :param float x: X coordinate, in millimeters
    :param float y: Y coordinate, in millimeters
    :param float z: Z coordinate, in millimeters
    """

    x: float
    y: float
    z: float

    @classmethod
    def from_tvec(cls, x: float, y: float, z: float):
        """
        Convert coordinate system to standard right-handed cartesian system.

        The pose estimation coordinate system has the origin at the camera center.
        Also converts units to millimeters.

        :param float x: The x-axis is to the right in the image taken by the camera.
        :param float y: The y-axis is down in the image taken by the camera.
        :param float z: The z-axis points from the camera center out the camera lens.
        """
        if os.environ.get('ZOLOTO_LEGACY_AXIS'):
            return cls(x=x * 1000, y=y * 1000, z=z * 1000)
        else:
            return cls(x=z * 1000, y=-x * 1000, z=-y * 1000)


class SphericalCoordinate(NamedTuple):
    """
    A 3 dimesional spherical coordinate location.

    The convential spherical coordinate in mathematical notation where θ is
    a rotation around the vertical axis and φ is measured as the angle from
    the vertical axis.
    More information: https://mathworld.wolfram.com/SphericalCoordinates.html

    :param float distance: Radial distance from the origin, in millimeters.
    :param float theta: Azimuth angle, θ, in radians. This is the angle from
        directly in front of the camera to the vector which points to the
        location in the horizontal plane. A positive value indicates a
        counter-clockwise rotation. Zero is at the centre of the image.
    :param float phi: Polar angle, φ, in radians. This is the angle "down"
        from the vertical axis to the vector which points to the location.
        Zero is directly upward.
    """

    distance: int
    theta: float
    phi: float

    @property
    def rot_x(self) -> float:
        """
        Rotation around the x-axis.

        Conventional:  This is unused.
        Legacy: A rotation up to down around the camera, in radians. Values
                increase as the marker moves towards the bottom of the image.
                A zero value is halfway up the image.
        """
        if os.environ.get('ZOLOTO_LEGACY_AXIS'):
            return self.phi - (pi / 2)
        else:
            raise AttributeError(
                "That axis is not available in the selected coordinate system.")

    @property
    def rot_y(self) -> float:
        """
        Rotation around the y-axis.

        Conventional: A rotation up to down around the camera, in radians.
                      Values increase as the marker moves towards the bottom
                      of the image. A zero value is halfway up the image.
        Legacy: A rotation left to right around the camera, in radians. Values
                increase as the marker moves towards the right of the image.
                A zero value is on the centerline of the image.
        """
        if os.environ.get('ZOLOTO_LEGACY_AXIS'):
            return -self.theta
        else:
            return self.phi - (pi / 2)

    @property
    def rot_z(self) -> float:
        """
        Rotation around the z-axis.

        Conventional: A rotation right to left around the camera, in radians.
                      Values increase as the marker moves towards the left of
                      the image. A zero value is on the centerline of the
                      image.
        Legacy: This is unused.
        """
        if os.environ.get('ZOLOTO_LEGACY_AXIS'):
            raise AttributeError(
                "That axis is not available in the selected coordinate system.")
        else:
            return self.theta

    @classmethod
    def from_tvec(cls, x: float, y: float, z: float):
        """
        Convert coordinate system to standard right-handed cartesian system.

        The pose estimation coordinate system has the origin at the camera center.

        :param float x: The x-axis is to the right in the image taken by the camera.
        :param float y: The y-axis is down in the image taken by the camera.
        :param float z: The z-axis points from the camera center out the camera lens.
        """
        _x, _y, _z = z, -x, -y

        dist = hypotenuse([_x, _y, _z])
        theta = atan2(_y, _x)
        phi = acos(_z / dist)
        return cls(int(dist * 1000), theta, phi)


ThreeTuple = Tuple[float, float, float]
RotationMatrix = Tuple[ThreeTuple, ThreeTuple, ThreeTuple]


class Orientation:
    """The orientation of an object in 3-D space."""

    __MARKER_ORIENTATION_CORRECTION = Quaternion(matrix=np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]))
    __ZOLOTO_LEGACY_ORIENTATION = Quaternion(matrix=np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ]))

    def __init__(self, rotation_matrix: NDArray, aruco_orientation: bool = True):
        """
        Construct a quaternion given the rotation matrix in the camera's coordinate system.

        More information:
        https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        # Calculate the quaternion of the rotation in the camera's coordinate system
        initial_rot = Quaternion(matrix=rotation_matrix)
        # remap quaternion to global coordinate system from the token's perspective
        quaternion = Quaternion(
            initial_rot.w, -initial_rot.z, -initial_rot.x, initial_rot.y,
        )
        if aruco_orientation:
            # Rotate the quaternion so 0 roll is a marker the correct way up
            quaternion *= self.__MARKER_ORIENTATION_CORRECTION

        self.__rotation_matrix = quaternion.rotation_matrix

        if os.environ.get('ZOLOTO_LEGACY_AXIS'):
            self._quaternion = initial_rot * self.__ZOLOTO_LEGACY_ORIENTATION
        else:
            self._quaternion = quaternion

        self._yaw_pitch_roll = quaternion.yaw_pitch_roll

    @property
    def rot_x(self) -> float:
        """
        Get rotation angle around X axis in radians.

        The roll rotation with zero as the April Tags marker reference point
        at the top left of the marker.

        Legacy: The inverted pitch rotation with zero as the marker facing
                directly away from the camera and a positive rotation being
                downward.
                The practical effect of this is that an April Tags marker
                facing the camera square-on will have a value of ``pi`` (or
                equivalently ``-pi``).
        """
        return self.yaw_pitch_roll[2]

    @property
    def rot_y(self) -> float:
        """
        Get rotation angle around Y axis in radians.

        The pitch rotation with zero as the marker facing the camera square-on
        and a positive rotation being upward.

        Legacy: The inverted yaw rotation with zero as the marker facing the
                camera square-on and a positive rotation being
                counter-clockwise.
        """
        return self.yaw_pitch_roll[1]

    @property
    def rot_z(self) -> float:
        """
        Get rotation angle around Z axis in radians.

        The yaw rotation with zero as the marker facing the camera square-on
        and a positive rotation being clockwise.

        Legacy: The roll rotation with zero as the marker facing the camera
                square-on and a positive rotation being clockwise.
        """
        return self.yaw_pitch_roll[0]

    @property
    def yaw(self) -> float:
        """
        Get yaw of the marker, a rotation about the vertical axis, in radians.

        Positive values indicate a rotation clockwise from the perspective of
        the marker.

        Zero values have the marker facing the camera square-on.
        """
        return self._yaw_pitch_roll[0]

    @property
    def pitch(self) -> float:
        """
        Get pitch of the marker, a rotation about the transverse axis, in radians.

        Positive values indicate a rotation upwards from the perspective of the
        marker.

        Zero values have the marker facing the camera square-on.
        """
        return self._yaw_pitch_roll[1]

    @property
    def roll(self) -> float:
        """
        Get roll of the marker, a rotation about the longitudinal axis, in radians.

        Positive values indicate a rotation clockwise from the perspective of
        the marker.

        Zero values have the marker facing the camera square-on.
        """
        return self._yaw_pitch_roll[2]

    @property
    def yaw_pitch_roll(self) -> ThreeTuple:
        """
        Get the equivalent yaw-pitch-roll angles.

        Specifically intrinsic Tait-Bryan angles following the z-y'-x'' convention.
        """
        return self._quaternion.yaw_pitch_roll

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        Get the rotation matrix represented by this orientation.

        Returns:
            A 3x3 rotation matrix as a tuple of tuples.
        """
        return self.__rotation_matrix.tolist()

    @property
    def quaternion(self) -> Quaternion:
        """Get the quaternion represented by this orientation."""
        return self._quaternion

    def __repr__(self) -> str:
        return "Orientation(rot_x={}, rot_y={}, rot_z={})".format(
            self.rot_x, self.rot_y, self.rot_z,
        )


class Marker:
    """Wrapper of a marker detection with axis and rotation calculated."""

    def __init__(
        self,
        marker: Detection,
        *,
        aruco_orientation: bool = True,
    ):
        self.marker = marker

        self.__marker_type = MarkerType(marker.tag_family.decode('utf-8'))
        self._id = marker.tag_id
        self.__pixel_center = PixelCoordinates(*marker.center.tolist())
        self._pixel_corners = marker.corners.tolist()
        self.__size = int(marker.tag_size * 1000) if marker.tag_size is not None else 0
        self._tvec = marker.pose_t
        self._rvec = marker.pose_R
        self.__aruco_orientation = aruco_orientation

        if self._tvec is not None:
            self.__distance = int(hypotenuse(self._tvec) * 1000)
        else:
            self.__distance = 0

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self.id} size={self.size} "
            f"type={self.marker_type.name} distance={self.distance}>")

    @property  # noqa: A003
    def id(self) -> int:  # noqa: A003
        """The marker id number."""
        return self._id

    @property
    def size(self) -> int:
        """The size of the detected marker in millimeters."""
        return self.__size

    @property
    def marker_type(self) -> MarkerType:
        """The family of the detected marker, likely tag36h11."""
        return self.__marker_type

    @property
    def pixel_corners(self) -> List[PixelCoordinates]:
        """The pixels of the corners of the marker in the image."""
        return [
            PixelCoordinates(x, y)
            for x, y in self._pixel_corners
        ]

    @property
    def pixel_centre(self) -> PixelCoordinates:
        """The pixel location of the center of the marker in the image."""
        return self.__pixel_center

    @property
    def distance(self) -> int:
        """The distance between the marker and camera, in millimeters."""
        if self._tvec is not None and self._rvec is not None:
            return self.__distance
        return 0

    @property
    def orientation(self) -> Orientation:
        """The marker's orientation."""
        if self._rvec is not None:
            return Orientation(self._rvec, aruco_orientation=self.__aruco_orientation)
        raise RuntimeError("This marker was detected with an uncalibrated camera")

    @property
    def spherical(self) -> SphericalCoordinate:
        """The spherical coordinates of the marker's location relative to the camera."""
        if self._tvec is not None:
            return SphericalCoordinate.from_tvec(*self._tvec.flatten().tolist())
        raise RuntimeError("This marker was detected with an uncalibrated camera")

    @property
    def cartesian(self) -> CartesianCoordinates:
        """The cartesian coordinates of the marker's location relative to the camera."""
        if self._tvec is not None:
            return CartesianCoordinates.from_tvec(*self._tvec.flatten().tolist())
        raise RuntimeError("This marker was detected with an uncalibrated camera")

    def as_dict(self) -> Dict[str, Any]:
        """The marker data as a dict."""
        marker_dict = {
            "id": self._id,
            "size": self.__size,
            "pixel_corners": self._pixel_corners,
        }
        if self._tvec is not None and self._rvec is not None:
            marker_dict.update(
                {"rvec": self._rvec.tolist(), "tvec": self._tvec.tolist()},
            )
        return marker_dict
