import argparse
import importlib

subcommands = [
    "annotate_image",
    "annotate_video",
    # "calibrate",
    "camera_benchmark",
    "live",
    "marker_benchmark",
    # "marker_generator",
    "vision_debug",
]


def build_argparser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)
    for command in subcommands:
        mod_name = f"{__package__}.{command}"
        importlib.import_module(mod_name).create_subparser(subparsers)

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
