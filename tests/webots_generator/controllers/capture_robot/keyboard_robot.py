from math import tan
from pathlib import Path

from controller import Keyboard, Supervisor
from webots_vision import Marker


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

    keyboard = Keyboard()
    keyboard.enable(1000 // 33)

    save_path = Path(__file__).parents[2] / "output"
    save_path.mkdir(exist_ok=True)
    i = 0
    while True:
        key = keyboard.getKey()
        if key == ord(' '):
            while (save_path / f'img-{i:03d}.png').exists():
                i += 1
            camera.saveImage(str(save_path / f'img-{i:03d}.png'), 100)
            print(f"Saved image {i:03d}")
        elif key == ord('Q'):
            print("Recognitions:")
            recognitions = camera.getRecognitionObjects()
            for recognition in recognitions:
                marker = Marker.from_webots_recognition(recognition)
                print(marker, marker.orientation)
        robot.step(200)


if __name__ == "__main__":
    main()
