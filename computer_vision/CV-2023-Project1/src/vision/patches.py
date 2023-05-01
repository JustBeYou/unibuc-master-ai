import dataclasses

import numpy
import cv2
import imutils
from typing import List, Optional


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


def extract_nth_rectangle(image: numpy.ndarray, n: int, m: int) -> Optional[numpy.ndarray]:
    """
    Returns the n-th polygon with m sides from a given image. Inspired by
    https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/.
    """
    blurred_image = cv2.GaussianBlur(image, (7, 7), 3)
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                            2)
    threshold_image = cv2.bitwise_not(threshold_image)

    contours = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    found = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != m:
            continue

        found += 1
        if found == n:
            return approx

    return None
