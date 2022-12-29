import argparse
import importlib

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
    print(version)


def build_argparser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)
    for command in subcommands:
        mod_name = f"{__package__}.{command}"
        importlib.import_module(mod_name).create_subparser(subparsers)

    version_parser = subparsers.add_parser("version", help="Print package version")
    version_parser.set_defaults(func=print_versions)

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if "func" in args:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
