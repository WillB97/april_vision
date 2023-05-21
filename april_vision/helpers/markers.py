"""Utilities to assist with marker operations."""
from typing import Dict, Iterable


def generate_marker_size_mapping(
    marker_sizes: Dict[Iterable[int], int],
) -> Dict[int, float]:
    tag_sizes: Dict[int, float] = {}

    for marker_ids, marker_size in marker_sizes.items():
        # Unroll generators to give direct lookup
        for marker_id in marker_ids:
            # Convert to meters
            tag_sizes[marker_id] = float(marker_size) / 1000

    return tag_sizes
