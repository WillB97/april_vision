from pathlib import Path
from math import hypot
from itertools import cycle

import pytest
import pymupdf

from april_vision import cli, Processor
from april_vision.frame_sources import ImageSource
from april_vision.cli.marker_generator.utils import VEC_DPI


mm_per_pixel = 25.4 / VEC_DPI


@pytest.mark.parametrize("tag_num,tag_size", zip(range(0, 250, 50), range(100, 200, 20)))
def test_processor(tmp_path: Path, tag_num: int, tag_size: int):
    filename_pdf = tmp_path / f"test_img_{tag_num}.pdf"
    filename = tmp_path / f"test_img_{tag_num}.png"
    cli.main([
        'marker_generator', 'SINGLE', '--marker_size', str(tag_size),
        '--range', str(tag_num), '--all_filename', str(filename_pdf),
    ])

    # convert pdf to png
    doc = pymupdf.open(str(filename_pdf))
    assert len(doc) == 1, "Should have exactly one page"
    page = doc[0]
    pix = page.get_pixmap()
    pix.save(str(filename))
    print(f"Saved to {filename}")

    image = ImageSource(filename).read()

    markers = Processor().see(frame=image)
    assert len(markers) == 1, "Should detect exactly one marker"
    [marker] = markers

    assert marker.id == tag_num, f"Marker ID should be {tag_num}"

    # figure out size of marker in pixels
    next_corners = cycle(marker.pixel_corners)
    _ = next(next_corners)
    corner_dists_px = [    # distance between corners
        hypot(corner.x - corner_next.x, corner.y - corner_next.y)
        for corner, corner_next in zip(marker.pixel_corners, next_corners)
    ]

    corner_dists = [dist * mm_per_pixel for dist in corner_dists_px]

    assert all(
        corner_dist == pytest.approx(tag_size, rel=0.5)
        for corner_dist in corner_dists
    ), f"All sides of the marker should be the same length. {corner_dists} != {tag_size}"
