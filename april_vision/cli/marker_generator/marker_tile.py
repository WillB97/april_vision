"""
Used to generate an image tile which can be customised.

These image tiles can be arranged on a page in the different modes.
"""
import numpy as np
import reportlab.graphics.shapes as rl_shapes
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors as rl_colors

from april_vision.cli.utils import ApriltagFamily, get_tag_family
from april_vision.marker import MarkerType

from .utils import Coord, VecCoord, get_reportlab_colour, mm_to_pixels


def generate_tag_array(tag_data: ApriltagFamily, tag_id: int) -> NDArray:
    """
    Generate a marker array for a given tag family and tag ID.

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

    for i in range(left_border_edge, right_border_edge + 1):
        tag[left_border_edge][i] = 0
        tag[i][left_border_edge] = 0
        tag[right_border_edge][i] = 0
        tag[i][right_border_edge] = 0

    # Fill in pixels, black if bit index is zero
    max_index = tag_data.nbits - 1
    for i, (x, y) in enumerate(tag_data.bits):
        binary = bool(int(tag_data.codes[tag_id]) & (1 << (max_index - i)))
        tag[y + border_edge][x + border_edge] = 255 if binary else 0

    return tag


def generate_tag_vectors(
    tag_data: ApriltagFamily,
    tag_id: int,
    marker_size: int,
) -> rl_shapes.Group:
    """
    Generate a reportlab graphic group for a given tag family and tag ID.

    Uses the tag family object to generate a marker, returns this data as a group of vector
    rectangles where each of the rectangles is 1 pixel of the marker.

    :param tag_data: The tag family to generate the marker from.
    :param tag_id: The ID of the tag to generate.
    :param marker_size: The size of the marker in pixels.
    """
    # Create a group to store the marker
    marker_group = rl_shapes.Group()
    cell_size = int(marker_size / tag_data.width_at_border)

    cell_rect = rl_shapes.Rect(
        0, 0,
        cell_size, cell_size,
        strokeOpacity=0,  # type: ignore[arg-type]
        fillColor=rl_colors.black,  # type: ignore[arg-type]
    )

    def add_cell(x: int, y: int, white: bool) -> None:
        # Copy the cell rectangle and update the position and colour
        cell = cell_rect.copy()

        # Convert the cell index to pixel positions
        cell.x = x * cell_size
        cell.y = y * cell_size
        cell.fillColor = rl_colors.white if white else rl_colors.black
        marker_group.add(cell)

    # Find position of origin in grid
    border_edge = int((tag_data.total_width / 2.0) - (tag_data.width_at_border / 2.0))

    # Compute the outer border edges
    left_border_edge = border_edge - 1
    right_border_edge = border_edge + tag_data.width_at_border

    # Draw black/white border lines
    # Normal has white as the outer border, reversed has black as the outer border
    for i in range(left_border_edge, right_border_edge + 1):
        # outer pixel
        add_cell(i, left_border_edge, not tag_data.reversed_border)
        add_cell(left_border_edge, i, not tag_data.reversed_border)
        add_cell(i, right_border_edge, not tag_data.reversed_border)
        add_cell(right_border_edge, i, not tag_data.reversed_border)

        # In the corner pixels only draw the outer border
        if i != left_border_edge and i != right_border_edge:
            # inner pixel
            add_cell(i, left_border_edge + 1, tag_data.reversed_border)
            add_cell(left_border_edge + 1, i, tag_data.reversed_border)
            add_cell(i, right_border_edge - 1, tag_data.reversed_border)
            add_cell(right_border_edge - 1, i, tag_data.reversed_border)

    # Fill in pixels, black if bit index is zero
    max_index = tag_data.nbits - 1
    for i, (x, y) in enumerate(tag_data.bits):
        binary = bool(int(tag_data.codes[tag_id]) & (1 << (max_index - i)))
        y_out = (tag_data.total_width - 1) - (y + border_edge)
        add_cell(x + border_edge, y_out, binary)

    return marker_group


class MarkerTile:
    """
    Used to generate an image tile which can be customised.

    These image tiles can be arranged on a page in the different modes.
    """

    def __init__(
        self,
        tag_data: ApriltagFamily,
        tag_id: int,
        marker_size: int,
        aruco_orientation: bool = False,
    ):
        """
        Generate a basic marker, no overlays, scaled to the correct size.

        The marker PIL.Image can be accessed via MarkerTile.image.
        """
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
        self.top_left = Coord(0, 0)
        self.bottom_right = Coord(self.image.width, self.image.height)

    def add_border_line(
        self,
        border_width: int,
        border_colour: str,
    ) -> None:
        """
        Add a line around the border of the marker.

        This changes the current marker design in place.
        """
        bordered_image = ImageOps.expand(
            self.image,
            border=border_width,
            fill=border_colour
        )

        self.image = bordered_image

        # Update the coords of where the marker is in the tile
        self.top_left = Coord(
            self.top_left.x + border_width,
            self.top_left.y + border_width,
        )
        self.bottom_right = Coord(
            self.bottom_right.x + border_width,
            self.bottom_right.y + border_width,
        )

    def add_centre_ticks(
        self,
        tick_width: int,
        tick_length: int,
        tick_colour: str,
    ) -> None:
        """
        Add tick lines half way along the border of the marker.

        This changes the current marker design in place.
        """
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
        """
        Add the ID number in the top left square of the white border.

        This changes the current marker design in place.
        """
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
        """
        Expand the marker by one marker square and add description text to this area.

        This changes the current marker design in place.
        """
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
        self.top_left = Coord(
            self.top_left.x + marker_square_size,
            self.top_left.y + marker_square_size,
        )
        self.bottom_right = Coord(
            self.bottom_right.x + marker_square_size,
            self.bottom_right.y + marker_square_size,
        )


class MarkerTileVector:
    """
    Used to generate an vector tile which can be customised.

    These vector tiles can be arranged on a page in the different modes.
    """

    def __init__(
        self,
        tag_data: ApriltagFamily,
        tag_id: int,
        marker_size: int,
        aruco_orientation: bool = False,
    ):
        """
        Generate a basic marker, no overlays, scaled to the correct size.

        The marker reportlab.graphics.shapes.Group can be accessed via
        MarkerTileVector.vectors.
        """
        self.tag_data = tag_data
        self.tag_id = tag_id

        # Calculate the overall marker size
        self.pixel_size = marker_size / tag_data.width_at_border

        # Generate marker graphic group
        self.vectors = generate_tag_vectors(tag_data, tag_id, mm_to_pixels(marker_size))

        if aruco_orientation:
            # Rotate by 180deg to match the aruco format
            self.vectors.rotate(180)

        bl_x, _bl_y, tr_x, _tr_y = self.vectors.getBounds()

        self.marker_width = abs(tr_x - bl_x)

    @property
    def marker_origin(self) -> VecCoord:
        """Return the origin of the marker."""
        # The marker origin originally starts at 0,0
        current_transform = self.vectors.transform
        # Ignore the rotation and scaling so just return the translation elements
        return VecCoord(current_transform[4], current_transform[5])

    @property
    def marker_centre(self) -> VecCoord:
        """Return the centre of the marker."""
        return VecCoord(
            self.marker_origin.x + self.marker_width / 2,
            self.marker_origin.y + self.marker_width / 2,
        )

    def set_marker_origin(self, x: float, y: float) -> None:
        """Set the origin of the marker."""
        origin_translation = (
            x - self.marker_origin.x,
            y - self.marker_origin.y,
        )
        self.vectors.translate(*origin_translation)

    def set_marker_centre(self, x: float, y: float) -> None:
        """Set the centre of the marker."""
        origin_translation = (
            x - self.marker_centre.x,
            y - self.marker_centre.y,
        )
        self.vectors.translate(*origin_translation)

    def add_border_line(
        self,
        border_width: int,
        border_colour: str,
    ) -> None:
        """
        Add a line around the border of the marker.

        This changes the current marker design in place.
        """
        half_border = border_width / 2
        border_rect = rl_shapes.Rect(
            -half_border, -half_border,
            self.marker_width + half_border, self.marker_width + half_border,
            strokeWidth=border_width,  # type: ignore[arg-type]
            strokeColor=get_reportlab_colour(border_colour),  # type: ignore[arg-type]
            fillOpacity=0,  # type: ignore[arg-type]
        )
        self.vectors.add(border_rect)

    def add_centre_ticks(
        self,
        tick_width: int,
        tick_length: int,
        tick_colour: str,
    ) -> None:
        """
        Add tick lines half way along the border of the marker.

        This changes the current marker design in place.
        """
        half_way = self.marker_width / 2

        # Top
        self.vectors.add(rl_shapes.Line(
            half_way, self.marker_width,
            half_way, self.marker_width - tick_length,
            strokeColor=get_reportlab_colour(tick_colour),
            strokeWidth=tick_width,
        ))

        # Left
        self.vectors.add(rl_shapes.Line(
            0, half_way,
            tick_length, half_way,
            strokeColor=get_reportlab_colour(tick_colour),
            strokeWidth=tick_width,
        ))

        # Bottom
        self.vectors.add(rl_shapes.Line(
            half_way, 0,
            half_way, tick_length,
            strokeColor=get_reportlab_colour(tick_colour),
            strokeWidth=tick_width,
        ))

        # Right
        self.vectors.add(rl_shapes.Line(
            self.marker_width, half_way,
            self.marker_width - tick_length, half_way,
            strokeColor=get_reportlab_colour(tick_colour),
            strokeWidth=tick_width,
        ))

    def add_id_number(
        self,
        font: str,
        text_size: int,
        text_colour: str,
    ) -> None:
        """
        Add the ID number in the top left square of the white border.

        This changes the current marker design in place.
        """
        # Add text to the image
        marker_square_size = mm_to_pixels(self.pixel_size)

        pos_x, pos_y = 0.0, 0.0

        # Draw tag ID number in corner of white boarder
        border_edge = int((self.tag_data.total_width / 2.0)
                          - (self.tag_data.width_at_border / 2.0)
                          )
        if self.tag_data.reversed_border:
            pos_x = (
                (marker_square_size * border_edge)
                + (marker_square_size // 2)
            )
        else:
            pos_x = (
                (marker_square_size * (border_edge))
                - (marker_square_size // 2)
            )

        # vertical center is approx 40% of text size above baseline
        pos_y = self.marker_width - pos_x - (text_size * 0.4)

        self.vectors.add(rl_shapes.String(
            pos_x, pos_y,
            f"{self.tag_id}",
            fontSize=text_size,
            fontName=font,
            textAnchor='middle',
            fillColor=get_reportlab_colour(text_colour)
        ))

    def add_description_border(
        self,
        description_format: str,
        font: str,
        text_size: int,
        text_colour: str,
        double_text: bool = False,
    ) -> None:
        """
        Expand the marker by one marker square and add description text to this area.

        This changes the current marker design in place.
        """
        marker_square_size = mm_to_pixels(self.pixel_size)

        description_text = description_format.format(
            marker_type=self.tag_data.name, marker_id=self.tag_id
        )

        pos_x, pos_x2 = 0, self.marker_width
        # vertical center is approx 40% of text size above baseline
        pos_y = - (marker_square_size // 2) - (text_size * 0.4)

        self.vectors.add(rl_shapes.String(
            pos_x, pos_y,
            description_text,
            fontSize=text_size,
            fontName=font,
            textAnchor='start',
            fillColor=get_reportlab_colour(text_colour)
        ))

        if double_text:
            self.vectors.add(rl_shapes.String(
                pos_x2, pos_y,
                description_text,
                fontSize=text_size,
                fontName=font,
                textAnchor='end',
                fillColor=get_reportlab_colour(text_colour)
            ))


if __name__ == '__main__':
    tag_data = get_tag_family(MarkerType.APRILTAG_36H11.value)

    # tile = MarkerTile(tag_data, 73, 100)
    # tile.add_border_line(1, "lightgrey")
    # tile.add_centre_ticks(1, 10, "lightgrey")
    # tile.add_id_number("calibri.ttf", 12, "lightgrey")
    # tile.add_description_border("{marker_type} {marker_id}", "calibri.ttf", 12, "lightgrey")

    # tile.image.show()

    from reportlab.graphics import renderPDF
    from reportlab.graphics.shapes import Drawing

    from .utils import PageSize

    tag0_tile = MarkerTileVector(tag_data, 0, 150)
    page = PageSize.A4.pixels

    d = Drawing(page.x, page.y)

    tag0_tile.add_border_line(3, 'lightgrey')
    tag0_tile.add_centre_ticks(3, 40, 'lightgrey')
    tag0_tile.add_id_number('Times-Roman', 55, 'lightgrey')
    tag0_tile.add_description_border('{marker_type} {marker_id}', 'Times-Roman', 55, 'black')

    # Center the marker on the page
    tag0_tile.set_marker_centre(page.x / 2, page.y / 2)

    d.add(tag0_tile.vectors)
    renderPDF.drawToFile(d, 'test0.pdf')
