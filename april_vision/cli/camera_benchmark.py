"""
Open cameras and measures the performance.

Measures the FPS for different resolutions.
"""
import argparse

# TODO Impliment this


def main(args: argparse.Namespace) -> None:
    """Measure the FPS for different resolutions."""
    raise NotImplementedError("Camera benchmark is not implemented.")


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Camera_benchmark command parser."""
    parser = subparsers.add_parser("camera_benchmark")

    parser.set_defaults(func=main)
