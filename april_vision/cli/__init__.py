"""The april_vision CLI."""
import argparse
import importlib
import logging

from april_vision._version import version

subcommands = [
    "tools",
    "annotate_image",
    "annotate_video",
    "calibrate",
    # "camera_benchmark",
    "live",
    # "marker_benchmark",
    "marker_generator",
    "vision_debug",
]


def build_argparser() -> argparse.ArgumentParser:
    """Load subparsers from available subcommands."""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)
    for command in subcommands:
        mod_name = f"{__package__}.{command}"
        importlib.import_module(mod_name).create_subparser(subparsers)

    parser.add_argument(
        '--version', action='version', version=version, help="Print package version")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")

    return parser


def setup_logger(debug: bool = False) -> None:
    """Output all loggers to console with custom format at level INFO or DEBUG."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # log from all loggers to stdout
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    if debug:
        root_logger.setLevel(logging.DEBUG)


def main() -> None:
    """CLI entry-point."""
    parser = build_argparser()
    args = parser.parse_args()
    setup_logger(debug=args.debug)

    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
