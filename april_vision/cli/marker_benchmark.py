"""
Provide a folder of images and iterate over the images doing marker detection.

Give and output of the results.
Can provide a list of the actual answers and get a result of the error.
"""
import argparse

# TODO impliment
# marker_benchmark
#     folder of test images
# 	provide list of actual answers to check against
#     give performance result


def main(args: argparse.Namespace) -> None:
    """Iterate over images doing marker detection."""
    raise NotImplementedError("Marker benchmark is not implemented.")


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Marker_benchmark command parser."""
    parser = subparsers.add_parser("marker_benchmark")

    parser.set_defaults(func=main)
