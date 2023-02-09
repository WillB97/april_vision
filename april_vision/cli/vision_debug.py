"""
Generate the debug images of the vision processing steps.

Provide an input image and generate the debug output
of the vision processing steps.
"""
import argparse
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List

import cv2
from PIL import Image
from pyapriltags import Detector

LOGGER = logging.getLogger(__name__)

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


@contextmanager
def pushd(new_dir: str) -> Generator[None, None, None]:
    """Enter a directory for the context and return to the previous on exiting."""
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def process_debug(preserve: bool = True, collage: bool = True, clean: bool = False) -> None:
    """Convert debug outputs to PNG and optionally collage the images."""
    new_files = []

    for index, file in enumerate(pnm_files):
        LOGGER.info(f"Generating PNG for {file}.")
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


def create_collage(files: List[str], out: str) -> None:
    """Create a collage of the debug images."""
    LOGGER.info("Generating debug collage.")
    img = Image.open(files[-1])
    width, height = img.size

    target_img = Image.new("RGB", (width*4, height*2))
    for k, png in enumerate(files):
        col, row = divmod(k, 4)
        img = Image.open(png)
        img.thumbnail((width, height))
        target_img.paste(img, (width*row, height*col))

    target_img.save(out)
    LOGGER.info("Generated debug collage.")


def main(args: argparse.Namespace) -> None:
    """Generate the debug images of the vision processing steps."""
    if not args.input_file.exists():
        LOGGER.fatal("Input file does not exist.")
        return

    detector = Detector(quad_decimate=1.0, debug=True)

    frame = cv2.imread(str(args.input_file), cv2.IMREAD_GRAYSCALE)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    # change directory around the debug process
    with pushd(args.output_dir):
        results = detector.detect(frame)
        LOGGER.info(f"Found {len(results)} {'marker' if len(results)==1 else 'markers'}")

        process_debug(
            preserve=not args.cleanup,
            collage=args.collage,
            clean=args.collage_only)


def create_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Vision_debug command parser."""
    parser = subparsers.add_parser(
        "vision_debug",
        description="Generate the debug images of the vision processing steps.",
        help="Generate the debug images of the vision processing steps.",
    )

    parser.add_argument("input_file", type=Path, help="The image to process.")
    parser.add_argument(
        "output_dir", type=Path, help="The directory to save the output files to.")
    parser.add_argument(
        '--collage', action='store_true',
        help="Generate the collage image 'all.png' of all the processing steps.")
    parser.add_argument(
        '--no-cleanup', action='store_false', dest='cleanup',
        help="Don't remove the interim PNM and PS files.")
    parser.add_argument(
        '--collage-only', action='store_true',
        help="Remove separate debug images, leaving only 'all.png'")

    parser.set_defaults(func=main)
