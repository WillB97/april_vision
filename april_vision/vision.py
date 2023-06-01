"""The top-level Processor class for marker detection and annotation."""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from pyapriltags import Detector

from .frame_sources import FrameSource
from .marker import Marker
from .utils import Frame, normalise_marker_text

LOGGER = logging.getLogger(__name__)


class Processor:
    """A pyapriltags wrapper for fiducial detection from frame sources."""

    capture_filter: Callable[[NDArray], NDArray]
    marker_filter: Callable[[List[Marker]], List[Marker]]
    detection_hook: Callable[[Frame, List[Marker]], None]

    def __init__(
        self,
        frame_source: Optional[FrameSource] = None,
        calibration: Optional[Tuple[float, float, float, float]] = None,
        tag_sizes: Union[float, Dict[int, float], None] = None,
        *,
        tag_family: str = 'tag36h11',
        threads: int = 4,
        quad_decimate: float = 2,
        aruco_orientation: bool = True,
        name: str = "Camera",
        vidpid: str = "",
        mask_unknown_size_tags: bool = False,
    ) -> None:
        self.capture_filter = lambda frame: frame
        self.marker_filter = lambda markers: markers
        self.detection_hook = lambda frame, markers: None

        self._aruco_orientation = aruco_orientation
        self.calibration = calibration
        self.name = name
        self.vidpid = vidpid

        if frame_source is None:
            frame_source = FrameSource()
        self._frame_source = frame_source

        if tag_sizes is None:
            tag_sizes = {}
        self.tag_sizes = tag_sizes
        self.mask_unknown_size_tags = mask_unknown_size_tags

        self.detector = Detector(
            families=tag_family,
            nthreads=threads,
            quad_decimate=quad_decimate,
        )

    def _capture(self, fresh: bool = True) -> Frame:
        """Get another frame from the underlying camera."""
        colour_frame = self._frame_source.read(fresh)

        # hook to allow modification of the captured frame
        colour_frame = self.capture_filter(colour_frame)

        return Frame.from_colour_frame(colour_frame)

    def _detect(self, frame: Frame) -> List[Marker]:
        """Locate fiducial markers in frame using pyapriltags."""
        if self.calibration is None:
            detections = self.detector.detect(frame.grey_frame)
        else:
            detections = self.detector.detect(
                frame.grey_frame,
                estimate_tag_pose=True,
                camera_params=self.calibration,
                tag_size=self.tag_sizes,
            )

        markers: List[Marker] = []
        for detection in detections:
            if self.mask_unknown_size_tags and isinstance(self.tag_sizes, dict):
                # remove detections of marker ids we don't have a size for
                if detection.tag_id not in self.tag_sizes:
                    continue
            markers.append(Marker.from_detection(
                detection,
                aruco_orientation=self._aruco_orientation,
            ))

        # hook to filter and modify markers
        markers = self.marker_filter(markers)

        # hook to extract a frame and its markers
        self.detection_hook(frame, markers)

        return markers

    def _annotate(
        self,
        frame: Frame,
        markers: List[Marker],
        line_thickness: int = 2,
        text_scale: float = 1,
    ) -> Frame:
        """
        Annotate marker borders and ids onto frame.

        The annotation is in-place.
        """
        for marker in markers:
            integer_corners = np.array(marker.pixel_corners, dtype=np.int32)
            marker_id = f"id={marker.id}"

            for frame_type in frame:
                cv2.polylines(
                    frame_type,
                    [integer_corners],
                    isClosed=True,
                    color=(0, 255, 0),  # Green (BGR)
                    thickness=line_thickness,
                )

                # Add square at marker origin corner
                if self._aruco_orientation:
                    origin_square = np.array([
                        # index 1 is the top-left corner in aruco
                        integer_corners[1] + np.array([3, 3]),
                        integer_corners[1] + np.array([3, -3]),
                        integer_corners[1] + np.array([-3, -3]),
                        integer_corners[1] + np.array([-3, 3]),
                    ], dtype=np.int32)
                else:
                    origin_square = np.array([
                        # index -1 is the top-left corner in apriltag
                        integer_corners[-1] + np.array([3, 3]),
                        integer_corners[-1] + np.array([3, -3]),
                        integer_corners[-1] + np.array([-3, -3]),
                        integer_corners[-1] + np.array([-3, 3]),
                    ], dtype=np.int32)

                cv2.polylines(
                    frame_type,
                    [origin_square],
                    isClosed=True,
                    color=(0, 0, 255),  # red
                    thickness=line_thickness,
                )

                marker_text_scale = text_scale * normalise_marker_text(marker)

                # Approximately center the text
                text_origin = np.array(marker.pixel_centre, dtype=np.int32)
                text_origin += np.array(
                    [-40 * marker_text_scale, 10 * marker_text_scale], dtype=np.int32)

                cv2.putText(
                    frame_type,
                    marker_id,
                    text_origin,
                    cv2.FONT_HERSHEY_DUPLEX,
                    marker_text_scale,
                    color=(255, 191, 0),  # deep sky blue
                    thickness=2,
                )

        return frame

    def _save(self, frame: Frame, name: Union[str, Path], colour: bool = True) -> None:
        """Save a frame to a file, selectable between colour and grayscale."""
        if colour:
            output_frame = frame.colour_frame
        else:
            output_frame = frame.grey_frame

        path = Path(name)
        if not path.suffix:
            LOGGER.warning("No file extension given, defaulting to jpg")
            path = path.with_suffix(".jpg")

        cv2.imwrite(str(path), output_frame)

    def capture(self) -> NDArray:
        """Get the raw image data from the camera."""
        return self._capture().colour_frame

    def see(self, *, frame: Optional[NDArray] = None) -> List[Marker]:
        """Get markers that the camera can see."""
        if frame is None:
            frames = self._capture()
        else:
            frames = Frame.from_colour_frame(frame)
        return self._detect(frames)

    def see_ids(self, *, frame: Optional[NDArray] = None) -> List[int]:
        """Get a list of visible marker IDs."""
        if frame is None:
            frames = self._capture()
        else:
            frames = Frame.from_colour_frame(frame)
        markers = self._detect(frames)
        return [marker.id for marker in markers]

    def save(
        self,
        name: Union[str, Path],
        *,
        frame: Optional[NDArray] = None,
    ) -> None:
        """Save an annotated image to a file."""
        if frame is None:
            frames = self._capture()
        else:
            frames = Frame.from_colour_frame(frame)
        markers = self._detect(frames)
        frames = self._annotate(
            frames,
            markers,
        )
        self._save(frames, name)

    def close(self) -> None:
        """Close the underlying capture device."""
        self._frame_source.close()

    def set_marker_sizes(
        self,
        tag_sizes: Union[float, Dict[int, float]],
    ) -> None:
        """
        Set the size of tags that are used by pyapriltags for pose estimation.

        Sizes are in meters.
        """
        self.tag_sizes = tag_sizes
