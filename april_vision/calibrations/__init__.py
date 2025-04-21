"""Packaged calibration files."""
import os
from pathlib import Path

calibrations = os.environ.get('OPENCV_CALIBRATIONS', '.').split(':')
calibrations.append(str(Path(__file__).parent))

extra_calibrations = calibrations.copy()
extra_calibrations.insert(1, str(Path(__file__).parent / 'extra_calibrations'))
