"""The april_vision CLI."""
import argparse
import importlib
import logging

from april_vision._version import version

subcommands = [
    "annotate_image",
    "annotate_video",
    "calibrate",
    "camera_benchmark",
    "live",
    "marker_benchmark",
    # "marker_generator",
    "vision_debug",
]


def print_versions(args):
    """Print library version."""
    print(version)  # noqa: T201


def build_argparser():
    """Load subparsers from available subcommands."""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)
    for command in subcommands:
        mod_name = f"{__package__}.{command}"
        importlib.import_module(mod_name).create_subparser(subparsers)

    version_parser = subparsers.add_parser("version", help="Print package version")
    version_parser.set_defaults(func=print_versions)

    return parser


def setup_logger(debug=False):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # log from all loggers to stdout
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    if debug:
        root_logger.setLevel(logging.DEBUG)


def main():
    """CLI entry-point."""
    setup_logger()
    parser = build_argparser()
    args = parser.parse_args()

    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
