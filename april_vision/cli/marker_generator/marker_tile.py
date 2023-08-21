import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps

from april_vision.cli.utils import ApriltagFamily, get_tag_family
from april_vision.marker import MarkerType

from .utils import coord, mm_to_pixels


def generate_tag_array(tag_data: ApriltagFamily, tag_id: int) -> NDArray:
    """
    Uses the tag family object to generate a marker, returns this data as a 2d numpy array
    where each of the cells is 1 pixel of the marker.
    """
    # Create grid of tag size
    dim = tag_data.total_width
    tag = np.ones((dim, dim), dtype=np.uint8) * 255

    # Find position of origin in grid
    border_edge = int((tag_data.total_width / 2.0) - (tag_data.width_at_border / 2.0))

    # Draw black/white boarder lines
    if tag_data.reversed_border:
        left_border_edge = border_edge - 1
        right_border_edge = border_edge + tag_data.width_at_border
    else:
        left_border_edge = border_edge
        right_border_edge = border_edge + tag_data.width_at_border - 1

    for i in range(left_border_edge, right_border_edge+1):
        tag[left_border_edge][i] = 0
        tag[i][left_border_edge] = 0
        tag[right_border_edge][i] = 0
        tag[i][right_border_edge] = 0

    # Fill in pixels, black if bit index is zero
    max_index = tag_data.nbits - 1
    for i, (x, y) in enumerate(tag_data.bits):
        binary = bool(tag_data.codes[tag_id] & (1 << (max_index - i)))
        tag[y+border_edge][x+border_edge] = 255 if binary else 0

    return tag


class MarkerTile:
    def __init__(
        self,
        tag_data: ApriltagFamily,
        tag_id: int,
        marker_size: int,
        aruco_orientation: bool = False,
    ):
        self.tag_data = tag_data
        self.tag_id = tag_id

        # Calculate the overall marker size
        self.pixel_size = marker_size / tag_data.width_at_border
        required_size = int(self.pixel_size * tag_data.total_width)

        # Generate marker image and resize
        tag_array = generate_tag_array(tag_data, tag_id)

        if aruco_orientation:
            # Rotate by 180deg to match the aruco format
            tag_array = np.rot90(tag_array, k=2)

        marker_image = Image.fromarray(tag_array)
        resized_image = marker_image.resize(
            (mm_to_pixels(required_size), mm_to_pixels(required_size)),
            resample=0
        )

        self.image = resized_image

        self.marker_width = self.image.width
        self.marker_height = self.image.height

        # Update the coords of where the marker is in the tile
        self.top_left = coord(0, 0)
        self.bottom_right = coord(self.image.width, self.image.height)

    def add_border_line(
        self,
        border_width: int,
        border_colour: str,
    ) -> None:
        bordered_image = ImageOps.expand(
            self.image,
            border=border_width,
            fill=border_colour
        )

        self.image = bordered_image

        # Update the coords of where the marker is in the tile
        self.top_left = coord(
            self.top_left.x + border_width,
            self.top_left.y + border_width,
        )
        self.bottom_right = coord(
            self.bottom_right.x + border_width,
            self.bottom_right.y + border_width,
        )

    def add_centre_ticks(
        self,
        tick_width: int,
        tick_length: int,
        tick_colour: str,
    ) -> None:
        img_size = self.image.size[0]
        image_draw = ImageDraw.Draw(self.image)

        line_start = (img_size // 2) - (tick_width // 2)

        # Top
        image_draw.line(
            [line_start, 0, line_start, tick_length],
            width=tick_width,
            fill=tick_colour,
        )

        # Left
        image_draw.line(
            [0, line_start, tick_length, line_start],
            width=tick_width,
            fill=tick_colour,
        )

        # Bottom
        image_draw.line(
            [line_start, img_size - tick_length, line_start, img_size],
            width=tick_width,
            fill=tick_colour,
        )

        # Right
        image_draw.line(
            [img_size - tick_length, line_start, img_size, line_start],
            width=tick_width,
            fill=tick_colour,
        )

    def add_id_number(
        self,
        font: str,
        text_size: int,
        text_colour: str,
    ) -> None:
        # Add text to the image
        marker_square_size = mm_to_pixels(self.pixel_size)

        # Draw tag ID number in corner of white boarder
        border_edge = int((self.tag_data.total_width / 2.0)
                          - (self.tag_data.width_at_border / 2.0)
                          )
        if self.tag_data.reversed_border:
            id_pos = (self.top_left.x
                      + (marker_square_size * border_edge)
                      + (marker_square_size // 2)
                      )
        else:
            id_pos = (self.top_left.x
                      + (marker_square_size * (border_edge - 1))
                      + (marker_square_size // 2)
                      )

        image_draw = ImageDraw.Draw(self.image)
        image_draw.text(
            (id_pos, id_pos),
            f"{self.tag_id}",
            fill=text_colour,
            anchor="mm",
            font=ImageFont.truetype(font, text_size),
        )

    def add_description_border(
        self,
        description_format: str,
        font: str,
        text_size: int,
        text_colour: str,
        double_text: bool = False,
    ) -> None:
        marker_square_size = mm_to_pixels(self.pixel_size)

        # Expand the tile to add a white border with a width of 1 marker square
        bordered_image = ImageOps.expand(
            self.image,
            border=marker_square_size,
            fill="white"
        )
        image_draw = ImageDraw.Draw(bordered_image)

        description_text = description_format.format(
            marker_type=self.tag_data.name, marker_id=self.tag_id
        )

        # Draw text outside the marker
        image_draw.text(
            (marker_square_size, bordered_image.height - (marker_square_size // 2)),
            description_text,
            fill=text_colour,
            anchor="lm",
            font=ImageFont.truetype(font, text_size),
        )

        if double_text:
            image_draw.text(
                (
                    bordered_image.width - marker_square_size,
                    bordered_image.height - (marker_square_size // 2)
                ),
                description_text,
                fill=text_colour,
                anchor="rm",
                font=ImageFont.truetype(font, text_size),
            )

        self.image = bordered_image

        # Update the coords of where the marker is in the tile
        self.top_left = coord(
            self.top_left.x + marker_square_size,
            self.top_left.y + marker_square_size,
        )
        self.bottom_right = coord(
            self.bottom_right.x + marker_square_size,
            self.bottom_right.y + marker_square_size,
        )


if __name__ == '__main__':
    tag_data = get_tag_family(MarkerType.APRILTAG_36H11.value)

    tile = MarkerTile(tag_data, 73, 100)
    tile.add_border_line(1, "lightgrey")
    tile.add_centre_ticks(1, 10, "lightgrey")
    tile.add_id_number("calibri.ttf", 12, "lightgrey")
    tile.add_description_border("{marker_type} {marker_id}", "calibri.ttf", 12, "lightgrey")

    tile.image.show()
