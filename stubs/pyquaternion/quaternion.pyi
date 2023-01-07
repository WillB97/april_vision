from _typeshed import Incomplete
from typing import Tuple, overload

import numpy as np
from numpy.typing import NDArray

ThreeTuple = Tuple[float, float, float]
FourTuple = Tuple[float, float, float, float]


class Quaternion:
    q: Incomplete
    @overload
    def __init__(self, *, matrix: NDArray[np.int8]) -> None: ...
    @overload
    def __init__(self, *, axis: Tuple[float, float, float], scalar: float) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...
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
    # def __eq__(self, other): ...
    def __neg__(self) -> 'Quaternion': ...
    def __abs__(self) -> float: ...
    # def __add__(self, other): ...
    # def __iadd__(self, other): ...
    # def __radd__(self, other): ...
    # def __sub__(self, other): ...
    # def __isub__(self, other): ...
    # def __rsub__(self, other): ...
    def __mul__(self, other: 'Quaternion') -> 'Quaternion': ...
    # def __imul__(self, other): ...
    # def __rmul__(self, other): ...
    # def __matmul__(self, other): ...
    # def __imatmul__(self, other): ...
    # def __rmatmul__(self, other): ...
    # def __div__(self, other): ...
    # def __idiv__(self, other): ...
    # def __rdiv__(self, other): ...
    # def __truediv__(self, other): ...
    # def __itruediv__(self, other): ...
    # def __rtruediv__(self, other): ...
    # def __pow__(self, exponent): ...
    # def __ipow__(self, other): ...
    # def __rpow__(self, other): ...
    # @property
    # def conjugate(self) -> 'Quaternion': ...
    # @property
    # def inverse(self) -> 'Quaternion': ...
    # @property
    # def norm(self) -> float: ...
    # @property
    # def magnitude(self) -> float: ...
    # @property
    # def normalised(self) -> 'Quaternion': ...
    # @property
    # def polar_unit_vector(self): ...
    # @property
    # def polar_angle(self) -> float: ...
    # @property
    # def polar_decomposition(self): ...
    # @property
    # def unit(self) -> 'Quaternion': ...
    # def is_unit(self, tolerance: float = ...): ...
    # def rotate(self, vector): ...
    # @classmethod
    # def exp(cls, q: 'Quaternion') -> 'Quaternion': ...
    # @classmethod
    # def log(cls, q: 'Quaternion') -> 'Quaternion': ...
    # @classmethod
    # def exp_map(cls, q, eta): ...
    # @classmethod
    # def sym_exp_map(cls, q, eta): ...
    # @classmethod
    # def log_map(cls, q, p): ...
    # @classmethod
    # def sym_log_map(cls, q, p): ...
    # @classmethod
    # def absolute_distance(cls, q0: 'Quaternion', q1: 'Quaternion'): ...
    # @classmethod
    # def distance(cls, q0: 'Quaternion', q1: 'Quaternion') -> float: ...
    # @classmethod
    # def sym_distance(cls, q0: 'Quaternion', q1: 'Quaternion') -> float: ...
    # @classmethod
    # def slerp(cls, q0: 'Quaternion', q1: 'Quaternion', amount: float = ...) -> 'Quaternion': ...
    # @classmethod
    # def intermediates(cls, q0: 'Quaternion', q1: 'Quaternion', n: int, include_endpoints: bool = ...) -> Generator['Quaternion', None, None]: ...
    # def derivative(self, rate) -> 'Quaternion': ...
    # def integrate(self, rate, timestep) -> None: ...
    @property
    def rotation_matrix(self) -> NDArray[np.int8]: ...
    # @property
    # def transformation_matrix(self) -> np.ndarray: ...
    @property
    def yaw_pitch_roll(self) -> Tuple[float, float, float]: ...
    # def get_axis(self, undefined=...): ...
    # @property
    # def axis(self): ...
    # @property
    # def angle(self) -> float: ...
    # @property
    # def degrees(self) -> float: ...
    # @property
    # def radians(self) -> float: ...
    # @property
    # def scalar(self) -> float: ...
    # @property
    # def vector(self): ...
    # @property
    # def real(self) -> float: ...
    # @property
    # def imaginary(self): ...
    @property
    def w(self) -> float: ...
    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...
    @property
    def z(self) -> float: ...
    # @property
    # def elements(self): ...
    # def __getitem__(self, index: int) -> float: ...
    # def __setitem__(self, index: int, value: float) -> None: ...
    # def __copy__(self) -> 'Quaternion': ...
    # def __deepcopy__(self, memo) -> 'Quaternion': ...
    # @staticmethod
    # def to_degrees(angle_rad: Optional[float]) -> Optional[float]: ...
    # @staticmethod
    # def to_radians(angle_deg: Optional[float]) -> Optional[float]: ...
