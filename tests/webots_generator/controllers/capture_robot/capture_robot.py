import csv
from math import tan
from pathlib import Path
from typing import NamedTuple

from controller import Supervisor

SAVE_PATH = Path(__file__).parents[3] / "test_data"


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
    with open(SAVE_PATH / "locations.csv") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            data.append(row)

    return [
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
            )
        for row in data
    ]


def save(robot, camera, i):
    robot.step(100)
    camera.saveImage(str(SAVE_PATH / f'img-{i:03d}.png'), 100)
    print(f"Saved image {i:03d}")
    robot.step(100)


def set_orientation(robot, yaw, pitch, roll):
    yaw_field = robot.getFromDef('yaw_transform').getField('rotation')
    pitch_field = robot.getFromDef('pitch_transform').getField('rotation')
    roll_field = robot.getFromDef('marker_transform').getField('rotation')

    roll_field.setSFRotation([-1, 0, 0, roll])
    pitch_field.setSFRotation([0, 1, 0, pitch])
    yaw_field.setSFRotation([0, 0, -1, yaw])


def main():
    robot = Supervisor()
    camera = robot.getDevice('camera')

    camera.enable(32)
    print(
        f"fx={(camera.getWidth() / 2) / tan(camera.getFov() / 2)}, "
        f"fy={(camera.getWidth() / 2) / tan(camera.getFov() / 2)}, "
        f"cx={camera.getWidth() // 2}, "
        f"cy={camera.getHeight() // 2}"
    )

    marker_transform = robot.getFromDef('marker_transform')
    marker_position = marker_transform.getField('translation')

    for marker_test in load_test_values():
        marker_position.setSFVec3f([
            marker_test.distance,
            marker_test.horizontal,
            marker_test.vertical,
        ])
        set_orientation(robot, marker_test.yaw, marker_test.pitch, marker_test.roll)
        save(robot, camera, marker_test.index)


if __name__ == "__main__":
    main()
