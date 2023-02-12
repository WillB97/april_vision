"""
Classes for marker detections and various axis representations.

Setting the environment variable ZOLOTO_LEGACY_AXIS uses the axis that were
used in zoloto<0.9.0. Otherwise the conventional right-handed axis is used
where x is forward, y is left and z is upward.
"""
import os
from enum import Enum
from math import acos, atan2, pi
from typing import Any, Dict, Iterator, List, NamedTuple, Tuple, cast

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
    A 3 dimensional cartesian coordinate in the standard right-handed cartesian system.

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

    :param float x: X coordinate, in millimeters
    :param float y: Y coordinate, in millimeters
    :param float z: Z coordinate, in millimeters
    """

    x: float
    y: float
    z: float

    @classmethod
    def from_tvec(cls, x: float, y: float, z: float) -> 'CartesianCoordinates':
        """
        Convert coordinate system to standard right-handed cartesian system.

        The pose estimation coordinate system has the origin at the camera center.
        Also converts units to millimeters.

        :param float x: The x-axis is to the right in the image taken by the camera.
        :param float y: The y-axis is down in the image taken by the camera.
        :param float z: The z-axis points from the camera center out the camera lens.
        """
        return cls(x=z * 1000, y=-x * 1000, z=-y * 1000)


class SphericalCoordinate(NamedTuple):
    """
    A 3 dimensional spherical coordinate location.

    # The conventional spherical coordinate in mathematical notation where θ is
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

    @classmethod
    def from_tvec(cls, x: float, y: float, z: float) -> 'SphericalCoordinate':
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


class Orientation(NamedTuple):
    """
    The orientation of an object in 3-D space.

    :param float yaw:   Get yaw of the marker, a rotation about the vertical axis, in radians.
                        Positive values indicate a rotation clockwise from the perspective of the marker.
                        Zero values have the marker facing the camera square-on.
    :param float pitch: Get pitch of the marker, a rotation about the transverse axis, in radians.
                        Positive values indicate a rotation upwards from the perspective of the marker.
                        Zero values have the marker facing the camera square-on.
    :param float roll:  Get roll of the marker, a rotation about the longitudinal axis, in radians.
                        Positive values indicate a rotation clockwise from the perspective of the marker.
                        Zero values have the marker facing the camera square-on.
    """

    @classmethod
    def from_rvec_matrix(cls, rotation_matrix: NDArray, aruco_orientation: bool = True) -> 'Orientation':
        """
        Calculate the yaw, pitch and roll given the rotation matrix in the camera's coordinate system.

        More information:
        https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        # Calculate the quaternion of the rotation in the camera's coordinate system
        initial_rot = Quaternion(matrix=rotation_matrix)
        # remap quaternion to global coordinate system from the token's perspective

        return cls.from_quaternion(initial_rot, aruco_orientation)

    @classmethod
    def from_quaternion(cls, quaternion: Quaternion, aruco_orientation: bool = True) -> 'Orientation':
        """
        Calculate the yaw, pitch and roll given the quaternion in the camera's coordinate system.

        More information:
        https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        # remap quaternion to global coordinate system from the token's perspective
        quaternion = Quaternion(
            quaternion.w, -quaternion.z, -quaternion.x, quaternion.y,
        )
        if aruco_orientation:
            # Rotate the quaternion so 0 roll is a marker the correct way up
            marker_orientation_correction = Quaternion(matrix=np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]))

            quaternion *= marker_orientation_correction

        obj = cls(*quaternion.yaw_pitch_roll)
        obj._quaternion = quaternion

        return obj

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        Get the rotation matrix represented by this orientation.

        Returns:
            A 3x3 rotation matrix as a tuple of tuples.
        """
        return cast(RotationMatrix, self._quaternion.rotation_matrix.tolist())

    @property
    def quaternion(self) -> Tuple[float, float, float, float]:
        """Get the quaternion represented by this orientation."""
        return self._quaternion.elements


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
