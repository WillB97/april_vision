import argparse

"""
marker_benchmark:
Provide a folder of images and itterate over the images doing marker detection
Give and output of the results
Can provide a list of the actual answers and get a result of the error
"""

# TODO impliment
# marker_benchmark
#     folder of test images
# 	provide list of actual answers to check against
#     give performance result


def main(args: argparse.Namespace):
    print("Not Implimented - Marker benchmark")


def create_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("marker_benchmark")

    parser.set_defaults(func=main)
