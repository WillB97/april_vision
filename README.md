# april_vision

A fiducial marker system used by Student Robotics.
Uses [april tag](https://april.eecs.umich.edu/software/apriltag) markers to provide detection, pose and distance estimation for these markers.

## Installation

This library requires OpenCV but the default installation does not install OpenCV. There are a few different versions of OpenCV with different install sizes, to install the default package without OpenCV, run the following command.

```bash
pip install april-vision
```

To install the lightweight headless version OpenCV install the library with the following command.

```bash
pip install april-vision[opencv]
```

If you want to perform some of the more advanced features of the CLI (live view of the camera) you need the full version of OpenCV, which can be installed with the following command.

```bash
pip install april-vision[cli]
```

If you need to run the calibration feature in the CLI you will need to install the `opencv-contrib-python` module. All the versions of OpenCV clash so you should only have one installed.

## Example

```python
from april_vision import Processor, USBCamera, calibrations, find_cameras

cameras = find_cameras(calibrations)

try:
    camera = cameras[0]
except IndexError:
    print("No cameras found")
    exit()

source = USBCamera.from_calibration_file(
    camera.index,
    camera.calibration,
    camera.vidpid
)

cam = Processor(
    source,
    tag_family='tag36h11',
    quad_decimate=2.0,
    tag_sizes=0.08,
    calibration=source.calibration
)

markers = cam.see_ids()
print(markers)
```

## Tools

When installed april_vision can be used on the command line providing a few useful tools. Each of the tools listed below contain help text on correct usage accessed via the `-h` argument.

```
april_vision annotate_image
    Annotate an image file with the detected markers
```
```
april_vision annotate_video
    Annotate a video file with the detected markers
```
```
april_vision calibrate
    Generate camera calibration using a ChArUco board
```
```
april_vision live
    Live camera demonstration with marker annotation
```
```
april_vision vision_debug
    Generate the debug images of the vision processing steps
```
