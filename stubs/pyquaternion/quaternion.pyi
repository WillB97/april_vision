from collections.abc import Generator
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from _typeshed import Incomplete

ThreeTuple = Tuple[float, float, float]
FourTuple = Tuple[float, float, float, float]


class Quaternion:
    q: Incomplete
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __hash__(self) -> int: ...
    @classmethod
    def random(cls) -> 'Quaternion': ...
    def __format__(self, formatstr: str) -> str: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __bool__(self) -> bool: ...
    def __nonzero__(self) -> bool: ...
    def __invert__(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __neg__(self) -> 'Quaternion': ...
    def __abs__(self) -> float: ...
    def __add__(self, other: object) -> 'Quaternion': ...
    def __iadd__(self, other: object) -> 'Quaternion': ...
    def __radd__(self, other: object) -> 'Quaternion': ...
    def __sub__(self, other: object) -> 'Quaternion': ...
    def __isub__(self, other: object) -> 'Quaternion': ...
    def __rsub__(self, other: object) -> 'Quaternion': ...
    def __mul__(self, other: object) -> 'Quaternion': ...
    def __imul__(self, other: object) -> 'Quaternion': ...
    def __rmul__(self, other: object) -> 'Quaternion': ...
    def __matmul__(self, other: object) -> 'Quaternion': ...
    def __imatmul__(self, other: object) -> 'Quaternion': ...
    def __rmatmul__(self, other: object) -> 'Quaternion': ...
    def __div__(self, other: object) -> 'Quaternion': ...
    def __idiv__(self, other: object) -> 'Quaternion': ...
    def __rdiv__(self, other: object) -> 'Quaternion': ...
    def __truediv__(self, other: object) -> 'Quaternion': ...
    def __itruediv__(self, other: object) -> 'Quaternion': ...
    def __rtruediv__(self, other: object) -> 'Quaternion': ...
    def __pow__(self, exponent: float) -> 'Quaternion': ...
    def __ipow__(self, other: object) -> 'Quaternion': ...
    def __rpow__(self, other: object) -> 'Quaternion': ...
    @property
    def conjugate(self) -> 'Quaternion': ...
    @property
    def inverse(self) -> 'Quaternion': ...
    @property
    def norm(self) -> float: ...
    @property
    def magnitude(self) -> float: ...
    @property
    def normalised(self) -> 'Quaternion': ...
    @property
    def polar_unit_vector(self) -> NDArray: ...
    @property
    def polar_angle(self) -> float: ...
    @property
    def polar_decomposition(self) -> Tuple[NDArray, float]: ...
    @property
    def unit(self) -> 'Quaternion': ...
    def is_unit(self, tolerance: float = ...) -> bool: ...
    def rotate(
        self,
        vector: Union[NDArray, 'Quaternion']
    ) -> Union[NDArray, 'Quaternion']: ...
    @classmethod
    def exp(cls, q: 'Quaternion') -> 'Quaternion': ...
    @classmethod
    def log(cls, q: 'Quaternion') -> 'Quaternion': ...
    @classmethod
    def exp_map(cls, q: 'Quaternion', eta: 'Quaternion') -> 'Quaternion': ...
    @classmethod
    def sym_exp_map(cls, q: 'Quaternion', eta: 'Quaternion') -> 'Quaternion': ...
    @classmethod
    def log_map(cls, q: 'Quaternion', p: 'Quaternion') -> 'Quaternion': ...
    @classmethod
    def sym_log_map(cls, q: 'Quaternion', p: 'Quaternion') -> 'Quaternion': ...
    @classmethod
    def absolute_distance(cls, q0: 'Quaternion', q1: 'Quaternion') -> float: ...
    @classmethod
    def distance(cls, q0: 'Quaternion', q1: 'Quaternion') -> float: ...
    @classmethod
    def sym_distance(cls, q0: 'Quaternion', q1: 'Quaternion') -> float: ...
    @classmethod
    def slerp(cls, q0: 'Quaternion', q1: 'Quaternion', amount: float = ...) -> 'Quaternion': ...
    @classmethod
    def intermediates(cls, q0: 'Quaternion', q1: 'Quaternion', n: int, include_endpoints: bool = ...) -> Generator['Quaternion', None, None]: ...
    def derivative(self, rate: NDArray) -> 'Quaternion': ...
    def integrate(self, rate: NDArray, timestep: float) -> None: ...
    @property
    def rotation_matrix(self) -> NDArray: ...
    @property
    def transformation_matrix(self) -> NDArray: ...
    @property
    def yaw_pitch_roll(self) -> Tuple[float, float, float]: ...
    def get_axis(self, undefined: NDArray = ...) -> NDArray: ...
    @property
    def axis(self) -> NDArray: ...
    @property
    def angle(self) -> float: ...
    @property
    def degrees(self) -> float: ...
    @property
    def radians(self) -> float: ...
    @property
    def scalar(self) -> float: ...
    @property
    def vector(self) -> NDArray: ...
    @property
    def real(self) -> float: ...
    @property
    def imaginary(self) -> NDArray: ...
    @property
    def w(self) -> float: ...
    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...
    @property
    def z(self) -> float: ...
    @property
    def elements(self) -> NDArray: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float) -> None: ...
    def __copy__(self) -> 'Quaternion': ...
    # def __deepcopy__(self, memo) -> 'Quaternion': ...
    @staticmethod
    def to_degrees(angle_rad: Optional[float]) -> Optional[float]: ...
    @staticmethod
    def to_radians(angle_deg: Optional[float]) -> Optional[float]: ...
