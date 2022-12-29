import argparse
import cv2
from april_vision import Camera

"""
live:
Opens the camera and does live marker detection
Can add overlays of fps, marker annotation, marker distance, etc..
Option to save the current view of the camera
"""

# TODO add these functionality
# live
# 	anotate
# 	fps
# 	distance
#   save button


def main(args: argparse.Namespace):
    cam = Camera(args.id, (1280, 720))  # TODO open camera in a smarter way

    while True:
        frame = cam._capture()
        markers = cam._detect(frame)

        if args.annotate:
            cam._annotate(frame, markers, text_scale=0.5, line_thickness=2)

        cv2.imshow('image', frame.colour_frame)

        button = cv2.waitKey(1) & 0xFF
        if button == ord('q'):
            break


def create_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("live")

    parser.add_argument("--id", type=int, required=True)

    parser.add_argument("--annotate", action='store_true')

    parser.set_defaults(func=main)
