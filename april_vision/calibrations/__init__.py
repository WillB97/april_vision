"""Packaged calibration files."""
import os

calibrations = os.environ.get('OPENCV_CALIBRATIONS', '.').split(':')
calibrations.append(os.path.dirname(__file__))
