import argparse

"""
camera_benchmark:
Opens cameras and measures the performance,
measures the fps for different resolutions
"""

# TODO Impliment this


def main(args: argparse.Namespace):
    print("Not Implimented - Camera benchmark")


def create_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("camera_benchmark")

    parser.set_defaults(func=main)
