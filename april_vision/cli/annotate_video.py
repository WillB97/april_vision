import argparse

"""
annotate_video:
Takes an input video, adds annotation to each frame and saves the video
"""

# TODO impliment


def main(args: argparse.Namespace):
    print("Not Implimented - Annotate video")


def create_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("annotate_video")

    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)

    parser.set_defaults(func=main)
