"""Camera calibration script."""
import argparse
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from april_vision.cli.utils import get_tag_family
from april_vision.marker import MarkerType
from april_vision.utils import Frame
from april_vision.vision import Marker, Processor

LOGGER = logging.getLogger(__name__)


class CalBoard:
    """Class used to represent a calibration board."""

    def __init__(
        self,
        rows: int,
        columns: int,
        marker_size: float,
        marker_type: str,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.marker_size = marker_size
        self.marker_type = marker_type

        self.total_markers = rows * columns
        self.marker_details = get_tag_family(marker_type)

    def corners_from_id(self, marker_id: int) -> List[Tuple[float, float, float]]:
        """
        Takes an input of a marker ID and returns the coordinates of the corners of the marker.

        The coordinates are 3D real world positions, top left of the board is 0,0,0.
        The Z coordinate of the board is always zero.
        The list of co-ords are in the order:
        bottom_left, bottom_right, top_right, top_left
        """
        marker_pixel_size = self.marker_size / self.marker_details.width_at_border

        row, column = divmod(marker_id, self.columns)
        row = self.rows - (row + 1)

        top_left_x = (column * self.marker_size) + (column * marker_pixel_size)
        top_left_y = (row * self.marker_size) + (row * marker_pixel_size)

        return [
            (top_left_x, top_left_y + self.marker_size, 0.0),
            (top_left_x + self.marker_size, top_left_y + self.marker_size, 0.0),
            (top_left_x + self.marker_size, top_left_y, 0.0),
            (top_left_x, top_left_y, 0.0),
        ]


def frame_capture(
    cap: cv2.VideoCapture,
    num: int,
) -> List[Frame]:
    """Capture frames to be used for calibration."""
    frames = []
    LOGGER.info("Press space to capture frame.")

    while True:
        ret, frame = cap.read()

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)
        if k == 27:  # Esc
            break
        elif k == 10 or k == 32:  # Enter or Space
            frames.append(Frame.from_colour_frame(frame))
            LOGGER.info("Frame captured!")
            if len(frames) == num:
                break

    return frames


def parse_detections(
    board_detections: List[Marker],
    board_design: CalBoard,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    """
    Pairs up 2D pixel corners with 3D real world co-ords.

    Takes the input of marker detections and the board design and outputs two lists
    where board_obj_points[i] pairs with board_img_points[i].
    """
    board_obj_points = []
    board_img_points = []

    for marker in board_detections:
        for pixel_corner, object_corner in zip(
            marker.pixel_corners,
            board_design.corners_from_id(marker.id)
        ):
            board_obj_points.append(object_corner)
            board_img_points.append((pixel_corner.x, pixel_corner.y))

    return board_obj_points, board_img_points


def main(args: argparse.Namespace) -> None:
    """Main function for calibrate command."""
    # Setup the camera
    video_dev = cv2.VideoCapture(args.index)

    video_dev.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution[0])
    video_dev.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution[1])

    if args.set_fps is not None:
        video_dev.set(cv2.CAP_PROP_FPS, args.set_fps)

    if args.set_codec is not None:
        video_dev.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*args.set_codec))

    # Capture the frames
    frames = frame_capture(video_dev, args.frame_count)
    video_dev.release()

    # Calculate the design of the cal board
    board = CalBoard(args.board[0], args.board[1], args.board[2], args.tag_family)
    min_required_detections = int(board.total_markers * args.valid_threshold / 100)

    # Detect the markers
    objectPoints = []
    imagePoints = []

    processor = Processor(aruco_orientation=False)

    for frame in frames:
        detections = processor._detect(frame)
        LOGGER.info(f'Detected {len(detections)} markers in frame')

        # only use the image if the number of detections were over the threshold
        if len(detections) >= min_required_detections:
            board_obj_points, board_img_points = parse_detections(detections, board)

            # dtype has to be float32 for cv2 calibration function
            objectPoints.append(np.array(board_obj_points, dtype=np.float32))
            imagePoints.append(np.array(board_img_points, dtype=np.float32))
        else:
            LOGGER.error('Discarding image due to low marker detection rate')

    # Get the actual dimensions of the captured frames
    width, height = frames[0].grey_frame.shape[::-1]

    # Calculate the camera calibration
    reproj_error, camera_matrix, dist_coeff, rvec, tvec = cv2.calibrateCamera(
        objectPoints,
        imagePoints,
        (width, height),
        None,
        None,
    )

    # Output calibration results
    LOGGER.info(f"> Avg reprojection error\n{reproj_error}")
    LOGGER.info(f"> Resolution\n{width}x{height}")
    LOGGER.info(f"> Camera matrix\n{camera_matrix}")
    LOGGER.info(f"> Distortion coefficients\n{dist_coeff}")

    write_cal_file(
        args.filename,
        args.frame_count,
        width,
        height,
        camera_matrix,
        dist_coeff,
        reproj_error,
        args.vidpid,
    )


def write_cal_file(
    cal_filename: str,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    camera_matrix: NDArray,
    dist_coeff: NDArray,
    avg_reprojection_error: float,
    vidpid: Optional[str] = None,
) -> None:
    """
    Write the calibration data to an XML file.

    This file can be loaded by the detect_cameras module.
    The file is also compatible with the OpenCV calibration module.
    """
    LOGGER.info("Generating calibration XML file")
    output_filename = cal_filename
    if not output_filename.lower().endswith(".xml"):
        output_filename += ".xml"

    file = cv2.FileStorage(output_filename, cv2.FILE_STORAGE_WRITE)

    calibrationDate = datetime.now().strftime("%a %d %b %Y %H:%M:%S")
    file.write("calibrationDate", calibrationDate)

    file.write("framesCount", frame_count)

    file.startWriteStruct("cameraResolution", cv2.FILE_NODE_SEQ)
    file.write("", frame_width)
    file.write("", frame_height)
    file.endWriteStruct()

    file.write("cameraMatrix", camera_matrix)
    file.write("dist_coeffs", dist_coeff)

    file.write("avg_reprojection_error", avg_reprojection_error)

    if vidpid is not None:
        # Wrap the str in quotes so it is outputted and loaded correctly
        file.write("vidpid", f'"{vidpid}"')

    file.release()


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Calibrate command parser."""
    parser = subparsers.add_parser(
        "calibrate",
        help="Generate camera calibration",
    )

    parser.add_argument(
        "--index",
        required=True,
        type=int,
        help="The camera index"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("WIDTH", "HEIGHT"),
        help="Force camera resolution"
    )
    parser.add_argument(
        '--set_fps',
        type=int,
        default=None,
        help="The FPS to set the camera to"
    )
    parser.add_argument(
        '--set_codec',
        type=str,
        default=None,
        help="4-character code of codec to set camera to (e.g. MJPG)"
    )

    parser.add_argument(
        "-n", "--frame_count",
        type=int,
        default=15,
        help="Number of frames to capture for calibration"
    )
    parser.add_argument(
        "--vidpid",
        type=str,
        default=None,
        help="VID and PID to be put on the calibration file"
    )

    parser.add_argument(
        "--valid_threshold",
        type=int,
        choices=range(0, 101),
        metavar="[0-100]",
        default=50,
        help=(
            "Percentage threshold of markers in board that need to be detected, "
            "image will be discarded if lower than this threshold"
        ),
    )
    parser.add_argument(
        "--board",
        required=True,
        type=int,
        nargs=3,
        metavar=("ROWS", "COLS", "MARKER_SIZE"),
        help="Specify the calibration board design"
    )
    parser.add_argument(
        '--tag_family',
        default=MarkerType.APRILTAG_36H11.value,
        choices=[
            MarkerType.APRILTAG_16H5.value,
            MarkerType.APRILTAG_25H9.value,
            MarkerType.APRILTAG_36H11.value,
        ],
        help="Set the marker family used in the calibration board, defaults to '%(default)s'",
    )

    parser.add_argument(
        "--filename",
        required=True,
        type=str,
        help="Filename of outputted calibration file"
    )

    parser.set_defaults(func=main)
