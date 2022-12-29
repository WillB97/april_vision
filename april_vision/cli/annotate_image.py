import argparse

"""
annotate_image:
Takes an input image, adds annotation and saves the image
"""

# TODO impliment


def main(args: argparse.Namespace):
    print("Not Implimented - Annotate image")


def create_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("annotate_image")

    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)

    parser.set_defaults(func=main)
