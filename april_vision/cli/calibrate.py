"""
A charuco calibration script.

From: https://gist.github.com/naoki-mizuno/d25cbc3c59228291cabe50529d70894c
"""
import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Iterable, List, Tuple

import cv2
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


def read_chessboards(
    frames: Iterable[NDArray],
    aruco_dict: Any,
    board: Any,
) -> Tuple[Any, Any, Any]:
    """Charuco base pose estimation."""
    all_corners = []
    all_ids = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            # ret is the number of detected corners
            if ret > 0:
                all_corners.append(c_corners)
                all_ids.append(c_ids)
        else:
            LOGGER.error("Failed!")

    imsize = gray.shape
    return all_corners, all_ids, imsize


def capture_camera(
    cap: cv2.VideoCapture,
    num: int = 1,
    mirror: bool = False,
    size: Tuple[int, int] = None,
) -> List[NDArray]:
    """Capture frames to be used for calibration."""
    frames = []
    LOGGER.info("Press space to capture frame.")

    while True:
        ret, frame = cap.read()

        if mirror is True:
            frame = cv2.flip(frame, 1)

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)
        if k == 27:  # Esc
            break
        elif k == 10 or k == 32:  # Enter or Space
            frames.append(frame)
            LOGGER.info("Frame captured!")
            if len(frames) == num:
                break

    return frames


def main(args: argparse.Namespace) -> None:
    """Charuco calibration."""
    try:
        import cv2.aruco  # type: ignore
    except ImportError:
        LOGGER.critical("Calibration requires the opencv-contrib-python package")
        sys.exit(1)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(8, 6, 0.03, 0.023, aruco_dict)

    dev_num = args.camera
    video_dev = cv2.VideoCapture(dev_num)

    if args.resolution:
        video_dev.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution[0])
        video_dev.set(cv2.CAP_PROP_FRAME_HEIGHT, args.resolution[1])

    frames = capture_camera(video_dev, args.frame_count)
    if len(frames) == 0:
        LOGGER.error("No frame captured")
        sys.exit(1)
    all_corners, all_ids, imsize = read_chessboards(frames, aruco_dict, board)
    ret, camera_matrix, dist_coeff, rvec, tvec = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, imsize, None, None,
    )

    LOGGER.info(f"> Camera matrix\n{camera_matrix}")
    LOGGER.info(f"> Distortion coefficients\n{dist_coeff}")

    width = int(video_dev.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT))
    LOGGER.info(f"> Resolution\n{width}x{height}")

    video_dev.release()

    LOGGER.info("Generating calibration XML file")
    output_filename = args.cal_filename
    if not args.cal_filename.lower().endswith(".xml"):
        output_filename += ".xml"

    file = cv2.FileStorage(args.cal_filename, cv2.FILE_STORAGE_WRITE)

    calibrationDate = datetime.now().strftime("%a %d %b %Y %H:%M:%S")
    file.write("calibrationDate", calibrationDate)

    file.write("framesCount", args.frame_count)

    file.startWriteStruct("cameraResolution", cv2.FILE_NODE_SEQ)
    file.write("", width)
    file.write("", height)
    file.endWriteStruct()

    file.write("cameraMatrix", camera_matrix)
    file.write("dist_coeffs", dist_coeff)

    if args.vidpid is not None:
        # Wrap the str in quotes so it is outputed and loaded correctly
        file.write("vidpid", f'"{args.vidpid}"')

    file.release()


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Calibrate command parser."""
    parser = subparsers.add_parser(
        "calibrate",
        help="Generate camera calibration",
        description="""
        Generate camera calibration using a ChArUco board.

        Generate a board with:
        https://calib.io/pages/camera-calibration-pattern-generator
        260x200mm, 6x8 squares, 30mm checkers.
        """)
    parser.add_argument("camera", type=int, help="The camera index")
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Force camera resolution")
    parser.add_argument(
        "-n", "--frame_count", type=int, default=15,
        help="Number of frames to use for calbration")
    parser.add_argument(
        "--vidpid", type=str, default=None,
        help="VID and PID to be put on the calibration file")
    parser.add_argument("cal_filename", type=str, help="Filename of output calibration file")

    parser.set_defaults(func=main)
