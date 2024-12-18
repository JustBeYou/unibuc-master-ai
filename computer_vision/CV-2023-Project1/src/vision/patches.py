import dataclasses
import typing
from typing import List

import numpy


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


def grid_patches(image: numpy.ndarray, columns: int, rows: int, line_thickness: int, margin: int) -> typing.Tuple[
    List[Patch], int, int]:
    w, h = image.shape[0], image.shape[1]
    vertical_lines, horizontal_lines = columns - 1, rows - 1

    w_patch = (w - 2 * margin - vertical_lines * line_thickness) // columns
    h_patch = (h - 2 * margin - horizontal_lines * line_thickness) // rows

    make_point = lambda x, y: Point(margin + x * (w_patch + line_thickness), margin + y * (h_patch + line_thickness))

    patches = []
    for i in range(columns):
        for j in range(rows):
            patches.append(Patch(make_point(i, j), make_point(i + 1, j + 1), line_thickness, i, j))

    return patches, w_patch + line_thickness, h_patch + line_thickness


def patches_list_to_matrix(patches_list: List[Patch], rows: int, columns: int) -> List[List[Patch]]:
    matrix = [[None for _ in range(columns)] for _ in range(rows)]

    for patch in patches_list:
        matrix[patch.y][patch.x] = patch

    return matrix
