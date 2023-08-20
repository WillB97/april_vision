from math import tan
from pathlib import Path

from controller import Supervisor

i = 0

SAVE_PATH = Path(__file__).parents[3] / "test_data"


def save(robot, camera):
    global i
    robot.step(100)
    camera.saveImage(str(SAVE_PATH / f'img-{i:03d}.png'), 100)
    print(f"Saved image {i:03d}")
    robot.step(100)
    i += 1


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

    set_orientation(robot, 0, 0, 0)

    marker_position.setSFVec3f([2, 0.25, 0.25])
    save(robot, camera)
    marker_position.setSFVec3f([2, 0, 0.25])
    save(robot, camera)
    marker_position.setSFVec3f([2, -0.25, 0.25])
    save(robot, camera)
    marker_position.setSFVec3f([2, 0.25, 0])
    save(robot, camera)
    marker_position.setSFVec3f([2, 0, 0])
    save(robot, camera)
    marker_position.setSFVec3f([2, -0.25, 0])
    save(robot, camera)
    marker_position.setSFVec3f([2, 0.25, -0.25])
    save(robot, camera)
    marker_position.setSFVec3f([2, 0, -0.25])
    save(robot, camera)
    marker_position.setSFVec3f([2, -0.25, -0.25])
    save(robot, camera)

    marker_position.setSFVec3f([1, 0, 0])
    set_orientation(robot, 0, 0, 0)
    save(robot, camera)
    set_orientation(robot, 0.7854, 0, 0)
    save(robot, camera)
    set_orientation(robot, -0.7854, 0, 0)
    save(robot, camera)
    set_orientation(robot, 0, 0.7854, 0)
    save(robot, camera)
    set_orientation(robot, 0, -0.7854, 0)
    save(robot, camera)
    set_orientation(robot, 0.7854, 0.7854, 0)
    save(robot, camera)
    set_orientation(robot, 0.7854, -0.7854, 0)
    save(robot, camera)
    set_orientation(robot, -0.7854, 0.7854, 0)
    save(robot, camera)
    set_orientation(robot, -0.7854, -0.7854, 0)
    save(robot, camera)
    set_orientation(robot, 0, 0, 0.7854)
    save(robot, camera)
    set_orientation(robot, 0.7854, 0, 0.7854)
    save(robot, camera)
    set_orientation(robot, -0.7854, 0, 0.7854)
    save(robot, camera)
    set_orientation(robot, 0, 0.7854, 0.7854)
    save(robot, camera)
    set_orientation(robot, 0, -0.7854, 0.7854)
    save(robot, camera)
    set_orientation(robot, 0.7854, 0.7854, 0.7854)
    save(robot, camera)
    set_orientation(robot, 0.7854, -0.7854, 0.7854)
    save(robot, camera)
    set_orientation(robot, -0.7854, 0.7854, 0.7854)
    save(robot, camera)
    set_orientation(robot, -0.7854, -0.7854, 0.7854)
    save(robot, camera)
    set_orientation(robot, 0, 0, -0.7854)
    save(robot, camera)
    set_orientation(robot, 0.7854, 0, -0.7854)
    save(robot, camera)
    set_orientation(robot, -0.7854, 0, -0.7854)
    save(robot, camera)
    set_orientation(robot, 0, 0.7854, -0.7854)
    save(robot, camera)
    set_orientation(robot, 0, -0.7854, -0.7854)
    save(robot, camera)
    set_orientation(robot, 0.7854, 0.7854, -0.7854)
    save(robot, camera)
    set_orientation(robot, 0.7854, -0.7854, -0.7854)
    save(robot, camera)
    set_orientation(robot, -0.7854, 0.7854, -0.7854)
    save(robot, camera)
    set_orientation(robot, -0.7854, -0.7854, -0.7854)
    save(robot, camera)


if __name__ == "__main__":
    main()
