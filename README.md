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
from april_vision.examples.camera import setup_cameras

# Markers 0-100 are 80mm in size
tag_sizes = {
    range(0, 100): 80
}

# Returns a dict of index and camera
cameras = setup_cameras(tag_sizes)

if len(cameras) == 0:
    print("No cameras found")

for name, cam in cameras.items():
    print(name)
    print(cam.see())
```

## Tools

When installed april_vision can be used on the command line providing the following list of useful tools. Each of the tools contain help text on correct usage accessed via the `-h` argument.

```bash
annotate_image
annotate_video
calibrate
live
marker_generator
vision_debug
tools
    family_details
    list_cameras
```
