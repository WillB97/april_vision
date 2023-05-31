"""
Classes for marker detections and various axis representations.
"""
from enum import Enum
from math import acos, atan2, cos, degrees, sin
from typing import NamedTuple, Optional, Tuple, cast

import numpy as np
from numpy.linalg import norm as hypotenuse
from numpy.typing import NDArray
from pyapriltags import Detection
from pyquaternion import Quaternion

ThreeTuple = Tuple[float, float, float]
RotationMatrix = Tuple[ThreeTuple, ThreeTuple, ThreeTuple]


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

    :param float r: Radial distance from the origin, in millimeters.
    :param float theta: Azimuth angle, θ, in radians. This is the angle from
        directly in front of the camera to the vector which points to the
        location in the horizontal plane. A positive value indicates a
        counter-clockwise rotation. Zero is at the centre of the image.
    :param float phi: Polar angle, φ, in radians. This is the angle "down"
        from the vertical axis to the vector which points to the location.
        Zero is directly upward.
    """

    r: int
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

        r = hypotenuse([_x, _y, _z])
        theta = atan2(_y, _x)
        phi = acos(_z / r)
        return cls(int(r * 1000), theta, phi)


class Orientation(NamedTuple):
    """
    The orientation of an object in 3-D space.

    :param float yaw:   Get yaw of the marker, a rotation about the vertical axis, in radians.
                        Positive values indicate a rotation clockwise from the perspective of
                        the marker.
                        Zero values have the marker facing the camera square-on.
    :param float pitch: Get pitch of the marker, a rotation about the transverse axis, in
                        radians.
                        Positive values indicate a rotation upwards from the perspective of
                        the marker.
                        Zero values have the marker facing the camera square-on.
    :param float roll:  Get roll of the marker, a rotation about the longitudinal axis,
                        in radians.
                        Positive values indicate a rotation clockwise from the perspective of
                        the marker.
                        Zero values have the marker facing the camera square-on.
    """

    yaw: float
    pitch: float
    roll: float

    @classmethod
    def from_rvec_matrix(
        cls,
        rotation_matrix: NDArray,
        aruco_orientation: bool = True
    ) -> 'Orientation':
        """
        Calculate yaw, pitch, roll given the rotation matrix in the camera's coordinate system.

        More information:
        https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """
        # Calculate the quaternion of the rotation in the camera's coordinate system
        initial_rot = Quaternion(matrix=rotation_matrix)

        return cls.from_quaternion(initial_rot, aruco_orientation)

    @classmethod
    def from_quaternion(
        cls,
        quaternion: Quaternion,
        aruco_orientation: bool = True
    ) -> 'Orientation':
        """
        Calculate yaw, pitch, roll given the quaternion in the camera's coordinate system.

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

        return obj

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        Get the rotation matrix represented by this orientation.

        Conversion calculation: https://w.wiki/6gbp

        Returns:
            A 3x3 rotation matrix as a tuple of tuples.
        """
        psi, theta, phi = self.yaw, self.pitch, self.roll

        A12 = -cos(theta) * sin(psi) + sin(phi) * sin(theta) * cos(psi)
        A13 = sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)
        A22 = cos(theta) * cos(psi) + sin(phi) * sin(theta) * sin(psi)
        A23 = -sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)

        matrix = (
            (cos(theta) * cos(psi), A12, A13),
            (cos(theta) * sin(psi), A22, A23),
            (-sin(theta), sin(phi) * cos(theta), cos(phi) * cos(theta)),
        )

        return cast(RotationMatrix, matrix)

    @property
    def quaternion(self) -> Tuple[float, float, float, float]:
        """
        Get the quaternion represented by this orientation.

        Conversion calculation: https://w.wiki/6gbq

        Returns:
            A 4-tuple hamiltonian quaternion.
        """
        psi_2, theta_2, phi_2 = self.yaw / 2, self.pitch / 2, self.roll / 2
        w = cos(phi_2) * cos(theta_2) * cos(psi_2) + sin(phi_2) * sin(theta_2) * sin(psi_2)
        i = sin(phi_2) * cos(theta_2) * cos(psi_2) - cos(phi_2) * sin(theta_2) * sin(psi_2)
        j = cos(phi_2) * sin(theta_2) * cos(psi_2) + sin(phi_2) * cos(theta_2) * sin(psi_2)
        k = cos(phi_2) * cos(theta_2) * sin(psi_2) - sin(phi_2) * sin(theta_2) * cos(psi_2)

        return (w, i, j, k)


PixelCorners = Tuple[PixelCoordinates, PixelCoordinates, PixelCoordinates, PixelCoordinates]


class Marker(NamedTuple):
    """
    Wrapper of a marker detection with axis and rotation calculated.
    """

    rvec: Optional[NDArray]
    tvec: Optional[NDArray]

    id: int
    size: int
    marker_type: MarkerType
    pixel_corners: PixelCorners
    pixel_centre: PixelCoordinates

    distance: float = 0
    # In degrees, increasing clockwise
    bearing: float = 0

    cartesian: CartesianCoordinates = CartesianCoordinates(0, 0, 0)
    spherical: SphericalCoordinate = SphericalCoordinate(0, 0, 0)
    orientation: Orientation = Orientation(0, 0, 0)

    aruco_orientation: bool = True

    @classmethod
    def from_detection(
        cls,
        marker: Detection,
        *,
        aruco_orientation: bool = True,
    ) -> 'Marker':
        _tag_size = int((marker.tag_size or 0) * 1000)

        _pixel_corners = tuple(
            PixelCoordinates(x, y)
            for x, y in marker.corners.tolist()
        )
        _pixel_centre = PixelCoordinates(*marker.center.tolist())

        if marker.pose_t is not None and marker.pose_R is not None:
            _distance = int(hypotenuse(marker.pose_t) * 1000)

            _cartesian = CartesianCoordinates.from_tvec(
                *marker.pose_t.flatten().tolist()
            )
            _bearing = degrees(atan2(-_cartesian.y, _cartesian.x))

            _spherical = SphericalCoordinate.from_tvec(
                *marker.pose_t.flatten().tolist()
            )

            _orientation = Orientation.from_rvec_matrix(
                marker.pose_R,
                aruco_orientation=aruco_orientation,
            )

            return cls(
                rvec=marker.pose_R,
                tvec=marker.pose_t,
                id=marker.tag_id,
                size=_tag_size,
                marker_type=MarkerType(marker.tag_family.decode('utf-8')),
                pixel_corners=cast(PixelCorners, _pixel_corners),
                pixel_centre=_pixel_centre,
                distance=_distance,
                bearing=_bearing,
                cartesian=_cartesian,
                spherical=_spherical,
                orientation=_orientation,
                aruco_orientation=aruco_orientation,
            )
        else:
            return cls(
                rvec=None,
                tvec=None,
                id=marker.tag_id,
                size=_tag_size,
                marker_type=MarkerType(marker.tag_family.decode('utf-8')),
                pixel_corners=cast(PixelCorners, _pixel_corners),
                pixel_centre=_pixel_centre,
                aruco_orientation=aruco_orientation,
            )

    def has_pose(self) -> bool:
        return (self.rvec is not None and self.tvec is not None)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self.id} distance={self.distance:.0f}mm "
            f"bearing={self.bearing:.0f}° size={self.size}mm type={self.marker_type.name}>"
        )
