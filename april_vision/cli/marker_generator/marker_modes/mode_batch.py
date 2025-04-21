"""Marker_generator subparser BATCH to generate a PDF with tiled and single markers."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from reportlab.pdfgen import canvas

from april_vision._version import __version__

from ..utils import PageSize
from . import mode_single, mode_tile

EXAMPLE_CONFIG = {
    "filename": "markers.pdf",
    "pages": [
        {
            "type": "SINGLE",
            "page_size": "A4",
            "range": "1-5",
            "marker_size": 150,
            "description_format": "{marker_type} {marker_id}",
        },
        {
            "type": "TILE",
            "page_size": "A3",
            "range": "10-15",
            "repeat": 6,
            "num_rows": 3,
            "num_columns": 2,
            "column_padding": 20,
            "row_padding": 20,
            "marker_size": 80,
            "description_format": "{marker_type} {marker_id}",
        }
    ]
}


def load_namespace(config: dict[str, Any], defaults: argparse.Namespace) -> argparse.Namespace:
    """Load a namespace with the defaults and the configuration."""
    namespace = deepcopy(defaults)
    for key, value in config.items():
        if key == "type":
            continue
        setattr(namespace, key, value)
    return namespace


def main(args: argparse.Namespace) -> None:
    """Generate a combined PDF of multiple markers using the provided configuration."""
    # Test if the configuration file exists
    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file {args.config} does not exist")

    # Generate dummy parser to extract argument defaults
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    mode_single.create_subparser(subparsers)
    mode_tile.create_subparser(subparsers)

    # Extract defaults, assuming there are no required arguments
    single_defaults = parser.parse_args(["SINGLE"])
    tile_defaults = parser.parse_args(["TILE"])
    del parser
    del subparsers

    # Load configuration
    with args.config.open() as config_file:
        config = json.load(config_file)

    # Validate minimal keys exist
    for key in ['filename', 'pages']:
        if key not in config.keys():
            raise ValueError(f"Configuration lacks required top-level key {key!r}")

    # Create the initial canvas
    combined_filename = config['filename']
    combined_pdf = canvas.Canvas(combined_filename, pagesize=PageSize.A4.value)
    combined_pdf.setAuthor(f"april_vision {__version__}")
    combined_pdf.setTitle(Path(config['filename']).stem)

    # Iterate over the configuration and generate the markers
    for entry, subconfig in enumerate(config['pages']):
        if "type" not in subconfig.keys():
            raise ValueError(f"Pages configuration entry {entry} lacks required key \"type\"")

        if subconfig['type'].upper() == "SINGLE":
            marker_conf = load_namespace(subconfig, single_defaults)
            mode_single.main(marker_conf, initial_canvas=combined_pdf)
        elif subconfig['type'].upper() == "TILE":
            marker_conf = load_namespace(subconfig, tile_defaults)
            mode_tile.main(marker_conf, initial_canvas=combined_pdf)
        else:
            raise ValueError(
                f"Type of pages configuration entry {entry} has invalid value "
                f"of {subconfig['type']}"
            )

    combined_pdf.save()


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_generator subparser BATCH to generate a PDF with tiled and single markers."""
    parser = subparsers.add_parser(
        "BATCH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate a combined PDF allowing multiple and single markers per page",
        epilog=(
            "The configuration file should be a JSON file with the following format:\n"
            f"{json.dumps(EXAMPLE_CONFIG, indent=4)}\n"
            "The available keys are the input options to either marker_generator "
            "SINGLE or TILE"
        ),
    )

    parser.add_argument(
        '--config',
        type=Path,
        help=(
            'Path to the JSON file containing the marker configuration following the format '
            'in the description'),
        required=True,
    )

    parser.set_defaults(func=main)
