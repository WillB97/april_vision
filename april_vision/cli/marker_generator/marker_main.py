import argparse

from .marker_modes import mode_cal, mode_image, mode_single, mode_tile


def main(args: argparse.Namespace) -> None:
    if args.mode == "SINGLE":
        mode_single.main(args)
    elif args.mode == "TILE":
        mode_tile.main(args)
    elif args.mode == "CAL_BOARD":
        mode_cal.main(args)
    elif args.mode == "IMAGE":
        mode_image.main(args)
