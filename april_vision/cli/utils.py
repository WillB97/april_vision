from typing import List, NamedTuple, Tuple

import pyapriltags


class ApriltagFamily(NamedTuple):
    ncodes: int
    codes: List[int]
    width_at_border: int
    total_width: int
    reversed_border: bool
    nbits: int
    bits: List[Tuple[int, int]]
    h: int
    name: str

    def __str__(self) -> str:
        data = "Marker family info:\n"
        data += f"    Tag family:       {self.name}\n"
        data += f"    Valid markers:    0-{self.ncodes - 1}\n"
        data += f"    Total width:      {self.total_width}\n"
        data += f"    Width at border:  {self.width_at_border}\n"
        data += f"    Reversed border:  {self.reversed_border}\n"
        data += f"    Num bits:         {self.nbits}\n"
        data += f"    Hamming distance: {self.h}\n"
        return data


def get_tag_family(family: str) -> ApriltagFamily:
    """
    Use the C-types in pyapriltags to get the tag family object.
    This object contains the required data to draw all the tags for a given family.
    """
    d = pyapriltags.Detector(families=family)
    raw_tag_data = d.tag_families[family].contents

    tag_data = ApriltagFamily(
        ncodes=raw_tag_data.ncodes,
        codes=[raw_tag_data.codes[i] for i in range(raw_tag_data.ncodes)],
        width_at_border=raw_tag_data.width_at_border,
        total_width=raw_tag_data.total_width,
        reversed_border=raw_tag_data.reversed_border,
        nbits=raw_tag_data.nbits,
        bits=[
            (raw_tag_data.bit_x[i], raw_tag_data.bit_y[i])
            for i in range(raw_tag_data.nbits)
        ],
        h=raw_tag_data.h,
        name=raw_tag_data.name.decode("utf-8"),
    )
    return tag_data


def parse_ranges(ranges: str) -> List[int]:
    """
    Parse a comma seprated list of numbers which may include ranges
    specified as hyphen-separated numbers.
    From https://stackoverflow.com/questions/6405208
    """
    result: List[int] = []
    for part in ranges.split(","):
        if "-" in part:
            a_, b_ = part.split("-")
            a, b = int(a_), int(b_)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return result
