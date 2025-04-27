"""
Provide a folder of images and iterate over the images doing marker detection.

Give and output of the results.
Can provide a list of the actual answers and get a result of the error.
"""
import argparse
import csv
import logging
from math import degrees
from pathlib import Path
from typing import List

from ..marker import Marker, MarkerType
from ..utils import Frame, load_calibration
from ..vision import Processor

LOGGER = logging.getLogger(__name__)

CSV_HEADER = [
    'File',
    'Detection Index',
    'Tag ID',
    'X (mm)',
    'Y (mm)',
    'Z (mm)',
    'Distance (mm)',
    'Theta (deg)',
    'Phi (deg)',
    'Yaw (deg)',
    'Pitch (deg)',
    'Roll (deg)',
]


def convert_marker_to_csv_row(marker: Marker, filename: Path, index: int = 0) -> List[float]:
    """Extract the relevant fields from a marker and return them as a list."""
    return [
        filename.name,
        index,
        marker.id,
        marker.cartesian.x,
        marker.cartesian.y,
        marker.cartesian.z,
        marker.distance,
        degrees(marker.spherical.theta),
        degrees(marker.spherical.phi),
        degrees(marker.orientation.yaw),
        degrees(marker.orientation.pitch),
        degrees(marker.orientation.roll),
    ]


def main(args: argparse.Namespace) -> None:
    """Iterate over images doing marker detection."""
    LOGGER.info(f"Assuming markers are {args.tag_family} and size {args.tag_size}mm")
    resolution, calibration = load_calibration(args.calibration)

    if not args.folder.exists() or not args.folder.is_dir():
        raise FileNotFoundError(f"Folder {args.folder} does not exist.")
    if not args.output.parent.exists():
        raise FileNotFoundError(f"Output folder {args.output.parent} does not exist.")

    if args.output.exists() and args.output.is_dir():
        args.output = args.output / "output.csv"

    if args.benchmark:
        if not args.benchmark.exists() or not args.benchmark.is_file():
            raise FileNotFoundError(f"Benchmark file {args.benchmark} does not exist.")

        with open(args.benchmark, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'File' not in reader.fieldnames:
                raise ValueError("Benchmark file does not contain 'File' column.")
            benchmark_data = {Path(row['File']): row for row in reader}

        # Only process images in the benchmark file
        images = [
            args.folder / filename
            for filename in benchmark_data.keys()
            if (args.folder / filename).exists()
        ]
        input_columns = [
            col for col in CSV_HEADER
            if col in reader.fieldnames and col not in ['File', 'Detection Index', 'Tag ID']
        ]
    else:
        # Get all images in the folder
        images = sorted(args.folder.iterdir())

    # Create processor without frame source
    processor = Processor(
        tag_family=args.tag_family,
        quad_decimate=args.quad_decimate,
        tag_sizes=float(args.tag_size) / 1000,
        calibration=calibration,
    )

    # iterate over images and output detections to csv
    with open(args.output, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        if args.benchmark:
            colnames = ['File', 'Tag ID']
            for col in input_columns:
                colnames.extend([col, f"Abs {col} Error", f"Rel {col} Error %"])
            for col in CSV_HEADER[3:]:
                if col not in colnames:
                    colnames.append(col)
            csv_writer.writerow(colnames)
        else:
            csv_writer.writerow(CSV_HEADER)

        for image in images:
            if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Load and process the image
            frame = Frame.from_file(image)
            frame_dims = frame.colour_frame.shape
            frame_res = (frame_dims[1], frame_dims[0])
            if frame_res != resolution:
                LOGGER.warning(
                    f"Image {image.name} has resolution {frame_res} "
                    f"but expected {resolution}."
                )

            markers = processor.see(frame=frame.colour_frame)

            LOGGER.info(f"Found {len(markers)} markers in {image.name}.")

            if args.benchmark:
                if len(markers) != 1:
                    LOGGER.warning(f"Image {image.name} does not have exactly 1 marker.")
                    continue

                [marker] = markers
                marker_data = dict(zip(CSV_HEADER, convert_marker_to_csv_row(marker, image)))
                expected_data = benchmark_data[image.relative_to(args.folder)]

                row = [marker_data['File'], marker_data['Tag ID']]
                for col in input_columns:
                    expected = float(expected_data[col])
                    measured = marker_data[col]

                    row.append(measured)
                    row.append(measured - expected)
                    row.append(f"{(measured - expected) / expected * 100:.3f}%")

                for col, value in marker_data.items():
                    if col in ['File', 'Tag ID', 'Detection Index']:
                        continue

                    # Only add the value if it's not already in the row
                    if col not in input_columns:
                        row.append(value)

                csv_writer.writerow(row)
            else:
                for idx, marker in enumerate(markers, start=1):
                    csv_writer.writerow(convert_marker_to_csv_row(marker, image, idx))


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Bulk_process command parser."""
    parser = subparsers.add_parser(
        "bulk_process",
        description="Process a folder of images and output results to CSV.",
        help="Process a folder of images and output results to CSV.",
    )

    parser.add_argument(
        "--folder",
        type=Path,
        help="Folder containing images to process.",
        required=True,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Camera calibration file.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file.",
        required=True,
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        type=Path,
        help="CSV file containing expected distances.",
    )

    parser.add_argument(
        '--tag_family', default=MarkerType.APRILTAG_36H11.value,
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to detect, defaults to 'tag36h11'")
    parser.add_argument(
        '--quad_decimate', type=float, default=2,
        help="Set the level of decimation used in the detection stage")
    parser.add_argument(
        '--tag_size', type=int, default=80, help="The size of markers in millimeters")

    parser.set_defaults(func=main)
