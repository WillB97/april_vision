import argparse

from .marker_modes import mode_single, mode_tile


def main(args: argparse.Namespace) -> None:
    if args.mode == "SINGLE":
        mode_single.main(args)
    if args.mode == "TILE":
        mode_tile.main(args)
