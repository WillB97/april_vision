import argparse

from .marker_modes import mode_single


def main(args: argparse.Namespace) -> None:
    if args.mode == "SINGLE":
        mode_single.main(args)
