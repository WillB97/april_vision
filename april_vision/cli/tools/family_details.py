import argparse
import logging

from april_vision.marker import MarkerType

from ..utils import get_tag_family

LOGGER = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    tag_data = get_tag_family(args.tag_family)
    print(tag_data)


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Print the details about a marker family."""
    parser = subparsers.add_parser(
        "family_details",
        description="Provide the details about a marker family",
        help="Provide the details about a marker family",
    )

    parser.add_argument(
        "tag_family",
        choices=[marker.value for marker in MarkerType],
        help="Set the marker family to display info about",
    )

    parser.set_defaults(func=main)
