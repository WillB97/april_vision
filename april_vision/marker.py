from enum import Enum
from math import acos, atan2
from typing import NamedTuple, List, Dict, Any, Tuple

import numpy as np
from numpy.linalg import norm as hypotenuse
from numpy.typing import NDArray
from pyquaternion import Quaternion
from pyapriltags import Detection


class MarkerType(Enum):
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
    :param float x: X coordinate
    :param float y: Y coordinate
    """

    x: float
    y: float


class CartesianCoordinates(NamedTuple):
    """
    A 3 dimesional cartesian coordinate in the standard right-handed cartesian system
    :param float x: X coordinate, positive is forward, in millimeters
    :param float y: Y coordinate, positive is left, in millimeters
    :param float z: Z coordinate, positive is up, in millimeters
    """

    x: float
    y: float
    z: float

    @classmethod
    def from_tvec(cls, x: float, y: float, z: float):
        """
        Convert coordinate system to standard right-handed cartesian system
        The pose estimation coordinate system has the origin at the camera center.

        :param float x: The x-axis is to the right in the image taken by the camera.
        :param float y: The y-axis is down in the image taken by the camera.
        :param float z: The z-axis points from the camera center out the camera lens.
        """
        return cls(z * 1000, -x * 1000, -y * 1000)


class SphericalCoordinate(NamedTuple):
    """
    :param float theta: the azimuthal angle in the xy-plane from the x-axis, in radians
    :param float phi: the polar angle from the z-axis, in radians
    :param float dist: Magnitude, in millimeters
    """

    dist: int
    theta: float
    phi: float

    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float):
        dist = hypotenuse([x, y, z])
        theta = atan2(y, x)
        phi = acos(z / dist)
        # cartesian coordinates are already in mm
        return cls(int(dist), theta, phi)


ThreeTuple = Tuple[float, float, float]
RotationMatrix = Tuple[ThreeTuple, ThreeTuple, ThreeTuple]


class Orientation:
    """The orientation of an object in 3-D space."""

    __MARKER_ORIENTATION_CORRECTION = Quaternion(matrix=np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]))

    def __init__(self, rotation_matrix: NDArray, aruco_orientation: bool = True):
        """
        Construct a quaternion given the rotation matrix in the camera's
        coordinate system.

        More information:
        https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        """

        # Calculate the quaternion of the rotation in the camera's coordinate system
        initial_rot = Quaternion(matrix=rotation_matrix)
        # remap quaternion to global coordinate system from the token's perspective
        quaternion = Quaternion(
            initial_rot.w, -initial_rot.z, -initial_rot.x, initial_rot.y
        )
        if aruco_orientation:
            # Rotate the quaternion so 0 roll is a marker the correct way up
            quaternion *= self.__MARKER_ORIENTATION_CORRECTION

        self.__rotation_matrix = quaternion.rotation_matrix
        self._quaternion = quaternion
        self._yaw_pitch_roll = quaternion.yaw_pitch_roll

    @property
    def rot_x(self) -> float:
        """Get rotation angle around x axis in radians."""
        return self.roll

    @property
    def rot_y(self) -> float:
        """Get rotation angle around y axis in radians."""
        return self.pitch

    @property
    def rot_z(self) -> float:
        """Get rotation angle around z axis in radians."""
        return self.yaw

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
        Get pitch of the marker, a rotation about the transverse axis, in
        radians.

        Positive values indicate a rotation upwards from the perspective of the
        marker.

        Zero values have the marker facing the camera square-on.
        """
        return self._yaw_pitch_roll[1]

    @property
    def roll(self) -> float:
        """
        Get roll of the marker, a rotation about the longitudinal axis, in
        radians.

        Positive values indicate a rotation clockwise from the perspective of
        the marker.

        Zero values have the marker facing the camera square-on.
        """
        return self._yaw_pitch_roll[2]

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
            self.rot_x, self.rot_y, self.rot_z
        )


class Marker:
    def __init__(
        self,
        marker: Detection,
        size: float,
        *,
        aruco_orientation: bool = True,
    ):
        self.marker = marker

        self.__marker_type = MarkerType(marker.tag_family.decode('utf-8'))
        self._id = marker.tag_id
        self.__pixel_center = PixelCoordinates(*marker.center.tolist())
        self._pixel_corners = marker.corners.tolist()
        self.__size = int(size * 1000)
        self._tvec = marker.pose_t
        self._rvec = marker.pose_R
        self.__pose = (self._tvec is not None and self._rvec is not None)
        self.__aruco_orientation = aruco_orientation

        self.__distance = int(hypotenuse(self._tvec)) if self.__pose else 0

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self.id} size={self.size} "
            f"type={self.marker_type.name} distance={self.distance}>")

    @property  # noqa: A003
    def id(self) -> int:  # noqa: A003
        return self._id

    @property
    def size(self) -> int:
        return self.__size

    @property
    def marker_type(self) -> MarkerType:
        return self.__marker_type

    @property
    def pixel_corners(self) -> List[PixelCoordinates]:
        return [
            PixelCoordinates(x, y)
            for x, y in self._pixel_corners
        ]

    @property
    def pixel_centre(self) -> PixelCoordinates:
        return self.__pixel_center

    @property
    def distance(self) -> int:
        return self.__distance

    @property
    def orientation(self) -> Orientation:
        if self.__pose:
            return Orientation(self._rvec, aruco_orientation=self.__aruco_orientation)
        return Orientation(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    @property
    def spherical(self) -> SphericalCoordinate:
        if self.__pose:
            return SphericalCoordinate.from_cartesian(*self.cartesian)
        return SphericalCoordinate(0, 0, 0)

    @property
    def cartesian(self) -> CartesianCoordinates:
        if self.__pose:
            return CartesianCoordinates.from_tvec(*self._tvec.flatten().tolist())
        return CartesianCoordinates(0, 0, 0)

    def as_dict(self) -> Dict[str, Any]:
        marker_dict = {
            "id": self._id,
            "size": self.__size,
            "pixel_corners": self._pixel_corners,
        }
        if self.__pose:
            marker_dict.update(
                {"rvec": self._rvec.tolist(), "tvec": self._tvec.tolist()}
            )
        return marker_dict
