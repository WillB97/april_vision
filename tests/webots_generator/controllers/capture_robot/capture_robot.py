import csv
from math import pi, tan, hypot, degrees
from pathlib import Path
from typing import NamedTuple

from pytest import approx

from controller import Supervisor
from webots_vision import Marker

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
    camera.recognitionEnable(32)
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

        hypot_dist = int(hypot(
            marker_test.horizontal,
            marker_test.vertical,
            marker_test.distance,
        ) * 1000)
        horz_angle = -marker_test.theta
        vert_angle = pi / 2 - marker_test.phi

        robot.step(50)
        recognitions = camera.getRecognitionObjects()
        assert len(recognitions) == 1, f"Found {len(recognitions)} markers"
        marker = Marker.from_webots_recognition(recognitions[0])
        print(marker, marker.orientation)

        assert marker.orientation.yaw == approx(marker_test.yaw), \
            f"Yaw: {marker.orientation.yaw} != {marker_test.yaw}"
        assert marker.orientation.pitch == approx(marker_test.pitch), \
            f"Pitch: {marker.orientation.pitch} != {marker_test.pitch}"
        assert marker.orientation.roll == approx(marker_test.roll), \
            f"Roll: {marker.orientation.roll} != {marker_test.roll}"
        assert marker.position.distance == approx(hypot_dist, abs=3), \
            f"Distance: {marker.position.distance} != {hypot_dist}"
        assert marker.position.horizontal_angle == approx(horz_angle, abs=degrees(1)), \
            f"Horizontal angle: {marker.position.horizontal_angle} != {horz_angle}"
        assert marker.position.vertical_angle == approx(vert_angle, abs=degrees(1)), \
            f"Vertical angle: {marker.position.vertical_angle} != {vert_angle}"

        save(robot, camera, marker_test.index)


if __name__ == "__main__":
    main()
