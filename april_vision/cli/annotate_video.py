"""
Annotate a video file.

Takes an input video, adds annotation to each frame and saves the video.
"""
import argparse
import logging
from pathlib import Path

import cv2

from ..frame_sources import VideoSource
from ..marker import MarkerType
from ..utils import annotate_text, load_calibration, normalise_marker_text
from ..vision import Processor

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Annotate an video file using the provided args."""
    input_file = Path(args.input_file)
    if not input_file.exists():
        LOGGER.fatal("Input file does not exist.")
        return

    source = VideoSource(args.input_file)
    num_frames = source._video.get(cv2.CAP_PROP_FRAME_COUNT)
    LOGGER.info(f"Processing video with {num_frames:.0f} frames.")

    fps = source._video.get(cv2.CAP_PROP_FPS)
    width = int(source._video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    calibration = None
    if args.calibration:
        _, calibration = load_calibration(args.calibration)

    processer = Processor(
        source,
        tag_family=args.tag_family,
        quad_decimate=args.quad_decimate,
        tag_sizes=float(args.tag_size) / 1000,
        calibration=calibration,
    )
    output = cv2.VideoWriter(
        args.output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while True:
        try:
            frame = processer._capture()
        except IOError:
            output.release()
            processer.close()
            break

        markers = processer._detect(frame)
        frame = processer._annotate(frame, markers)

        if args.calibration:
            try:
                for marker in markers:
                    # Check we have pose data
                    _ = marker.cartesian

                    text_scale = normalise_marker_text(marker)

                    loc = (
                        int(marker.pixel_centre.x - 80 * text_scale),
                        int(marker.pixel_centre.y + 40 * text_scale),
                    )
                    frame = annotate_text(
                        frame, f"dist={marker.distance}mm", loc,
                        text_scale=0.8 * text_scale,
                        text_colour=(255, 191, 0),  # deep sky blue
                    )
            except RuntimeError:
                pass

        # save frame
        output.write(frame.colour_frame)

        if args.preview:
            cv2.imshow('image', frame.colour_frame)
            _ = cv2.waitKey(1)

    LOGGER.info("Finished processing video.")
    if args.preview:
        cv2.destroyAllWindows()


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Annotate_video command parser."""
    parser = subparsers.add_parser(
        "annotate_video",
        description="Annotate a video file with its markers.",
        help="Annotate a video file with its markers.",
    )

    parser.add_argument("input_file", type=str, help="The video file to process.")
    parser.add_argument("output_file", type=str, help="The filepath to save the output to.")

    parser.add_argument(
        '--preview', action='store_true',
        help="Display the annotated video as it is annotated.")

    parser.add_argument(
        '--tag_family', default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")
    parser.add_argument(
        '--quad_decimate', type=float, default=2,
        help="Set the level of decimation used in the detection stage")

    parser.add_argument(
        '--calibration', type=Path, default=None, help="Calbration XML file to use.")
    parser.add_argument(
        '--tag_size', type=int, default=0, help="The size of markers in millimeters")

    parser.set_defaults(func=main)
