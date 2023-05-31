"""
Annotate an image file.

Takes an input image, adds annotation and saves the image.
"""
import argparse
import logging
from pathlib import Path

import cv2

from ..frame_sources import ImageSource
from ..marker import MarkerType
from ..utils import annotate_text, load_calibration, normalise_marker_text
from ..vision import Processor

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Annotate an image file using the provided args."""
    input_file = Path(args.input_file)
    if not input_file.exists():
        LOGGER.fatal("Input file does not exist.")
        return

    calibration = None
    if args.calibration:
        _, calibration = load_calibration(args.calibration)

    source = ImageSource(args.input_file)
    processer = Processor(
        source,
        tag_family=args.tag_family,
        quad_decimate=args.quad_decimate,
        tag_sizes=float(args.tag_size) / 1000,
        calibration=calibration,
    )

    frame = processer._capture()
    markers = processer._detect(frame)
    LOGGER.info(f"Found {len(markers)} markers.")
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
    cv2.imwrite(args.output_file, frame.colour_frame)

    if args.preview:
        cv2.imshow('image', frame.colour_frame)
        LOGGER.info("Press any key to close window.")
        _ = cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Annotate_image command parser."""
    parser = subparsers.add_parser(
        "annotate_image",
        description="Annotate an image file with its markers",
        help="Annotate an image file with its markers",
    )

    parser.add_argument("input_file", type=str, help="The image to process.")
    parser.add_argument("output_file", type=str, help="The filepath to save the output to.")

    parser.add_argument('--preview', action='store_true', help="Display the annotated image.")

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
