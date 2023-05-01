import dataclasses

import numpy
from typing import List


@dataclasses.dataclass
class Point:
    x: int
    y: int

    def as_tuple(self) -> tuple:
        return self.x, self.y


@dataclasses.dataclass
class Patch:
    top_left: Point
    bottom_right: Point
    line_thickness: int
    x: int
    y: int


def grid_patches(image: numpy.ndarray, columns: int, rows: int, line_thickness: int, margin: int) -> List[Patch]:
    w, h = image.shape[0], image.shape[1]
    vertical_lines, horizontal_lines = columns - 1, rows - 1

    w_patch = (w - 2 * margin - vertical_lines * line_thickness) // columns
    h_patch = (h - 2 * margin - horizontal_lines * line_thickness) // rows

    make_point = lambda x, y: Point(margin + x * (w_patch + line_thickness), margin + y * (h_patch + line_thickness))

    patches = []
    for i in range(columns):
        for j in range(rows):
            patches.append(Patch(make_point(i, j), make_point(i + 1, j + 1), line_thickness, i, j))

    return patches
