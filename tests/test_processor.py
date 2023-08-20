import csv
from math import radians
from pathlib import Path
from typing import NamedTuple

import pytest

from april_vision import Processor, Frame

CALIBRATION = [
    1293.0912575063312,  # fx
    1293.0912575063312,  # fy
    400,  # cx
    400,  # cy
]


@pytest.fixture
def processor():
    yield Processor(calibration=CALIBRATION, tag_sizes=0.2, aruco_orientation=False)


class MarkerValues(NamedTuple):
    index: int
    distance: float
    horizontal: float
    vertical: float
    theta: float
    phi: float
    r: float
    yaw: float
    pitch: float
    roll: float
    name: str = "marker"


def load_test_values():
    data = []
    with open(Path(__file__).parent / "test_data/locations.csv") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            data.append(row)

    return [
        pytest.param(
            MarkerValues(
                index=int(row["Index"]),
                distance=float(row["distance"]),
                horizontal=float(row["horizontal"]),
                vertical=float(row["vertical"]),
                theta=float(row["theta"]),
                phi=float(row["phi"]),
                r=float(row["r"]),
                yaw=float(row["yaw"]),
                pitch=float(row["pitch"]),
                roll=float(row["roll"]),
                name=row.get("name", "marker"),
            ),
            id=row.get("name", "marker"),
        )
        for row in data
    ]


@pytest.mark.parametrize("test_values", load_test_values())
def test_processor(processor: Processor, test_values: MarkerValues):
    frame = Frame.from_file(
        Path(__file__).parent / f"test_data/img-{test_values.index:03d}.png")

    markers = processor._detect(frame)
    assert len(markers) == 1, "Should only detect one marker"
    [marker] = markers

    assert marker.id == 14, "Marker ID should be 14"
    assert marker.size == 200, "Marker size should be 200"

    assert marker.cartesian.x == pytest.approx(test_values.distance * 1000, rel=1e-2, abs=5), (
        f"Distance to marker plane of {test_values.name} is incorrect. "
        f"{marker.cartesian.x} != {test_values.distance * 1000}")
    assert marker.cartesian.y == pytest.approx(
        test_values.horizontal * 1000, rel=1e-2, abs=5), (
        f"Horizontal position of {test_values.name} is incorrect. "
        f"{marker.cartesian.y} != {test_values.horizontal * 1000}")
    assert marker.cartesian.z == pytest.approx(test_values.vertical * 1000, rel=1e-2, abs=5), (
        f"Vertical position of {test_values.name} is incorrect. "
        f"{marker.cartesian.z} != {test_values.vertical * 1000}")

    assert marker.spherical.theta == pytest.approx(test_values.theta, abs=radians(5)), (
        f"Theta of {test_values.name} is incorrect. "
        f"{marker.spherical.theta} != {test_values.theta}")
    assert marker.spherical.phi == pytest.approx(test_values.phi, abs=radians(5)), (
        f"Phi of {test_values.name} is incorrect. "
        f"{marker.spherical.phi} != {test_values.phi}")
    assert marker.spherical.r == pytest.approx(test_values.r * 1000, rel=1e-2, abs=5), (
        f"Hypotenuse distance of {test_values.name} is incorrect. "
        f"{marker.spherical.r} != {test_values.r * 1000}")

    assert marker.orientation.yaw == pytest.approx(test_values.yaw, abs=radians(5)), (
        f"Yaw of {test_values.name} is incorrect. "
        f"{marker.orientation.yaw} != {test_values.yaw}")
    assert marker.orientation.pitch == pytest.approx(test_values.pitch, abs=radians(5)), (
        f"Pitch of {test_values.name} is incorrect. "
        f"{marker.orientation.pitch} != {test_values.pitch}")
    assert marker.orientation.roll == pytest.approx(test_values.roll, abs=radians(5)), (
        f"Roll of {test_values.name} is incorrect. "
        f"{marker.orientation.roll} != {test_values.roll}")
