"""Packaged calibration files."""
import os

extra_calibrations = os.environ.get('OPENCV_CALIBRATIONS', '.').split(':')
extra_calibrations.append(os.path.dirname(__file__))
