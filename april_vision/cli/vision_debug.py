import os
import cv2
import argparse
from PIL import Image
from pyapriltags import Detector

"""
vision_debug:
Provide an input image and generate the debug output
of the vision processing steps
"""


pnm_files = [
    "debug_preprocess.pnm",
    "debug_threshold.pnm",
    "debug_segmentation.pnm",
    "debug_clusters.pnm",
    "debug_quads_raw.pnm",
    "debug_quads_fixed.pnm",
    "debug_samples.pnm",
    "debug_output.pnm",
]

ps_files = [
    "debug_lines.ps",
    "debug_output.ps",
    "debug_quads.ps",
]


def process_debug(preserve=True, collage=True, clean=False):
    new_files = []

    for index, file in enumerate(pnm_files):
        with Image.open(file) as im:
            new_filename = f"{index}_{file.split('.')[0]}.png"
            im.save(new_filename)
            new_files.append(new_filename)

    if not preserve:
        for file in (ps_files + pnm_files):
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    if collage:
        create_collage(new_files, 'all.png')

        if clean:
            for file in new_files:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass


def create_collage(files, out):
    img = Image.open(files[-1])
    width, height = img.size

    target_img = Image.new("RGB", (width*4, height*2))
    for k, png in enumerate(files):
        col, row = divmod(k, 4)
        img = Image.open(png)
        img.thumbnail((width, height))
        target_img.paste(img, (width*row, height*col))

    target_img.save(out)


def main(args: argparse.Namespace):
    detector = Detector(quad_decimate=1.0, debug=True)

    # TODO change directory around the debug process
    frame = cv2.imread(args.input_file, cv2.IMREAD_GRAYSCALE)
    results = detector.detect(frame)
    print(f"Found {len(results)} {'marker' if len(results)==1 else 'markers'}")

    process_debug(preserve=False, collage=True, clean=True)


def create_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser("vision_debug")

    parser.add_argument("input_file", type=str)

    parser.set_defaults(func=main)
