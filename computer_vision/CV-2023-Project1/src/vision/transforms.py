import cv2
import numpy

from . import patches
from typing import List


def read(path: str) -> numpy.ndarray:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def scale(image: numpy.ndarray, factor: float) -> numpy.ndarray:
    w, h = int(image.shape[0] * factor), int(image.shape[1] * factor)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def grayscale(image: numpy.ndarray) -> numpy.ndarray:
    if len(image.shape) > 2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def colored_bgr(image: numpy.ndarray) -> numpy.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def draw_patches(image: numpy.ndarray, patches_list: List[patches.Patch]) -> numpy.ndarray:
    image = image.copy()

    for patch in patches_list:
        cv2.rectangle(
            image,
            patch.top_left.as_tuple(),
            patch.bottom_right.as_tuple(),
            (255, 0, 0),
            patch.line_thickness
        )

    return image


def filter_for_domino_circles(image: numpy.ndarray) -> numpy.ndarray:
    board = grayscale(image)
    blur = cv2.GaussianBlur(board, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh, thresh)
    kernel = numpy.ones((3, 3), dtype=numpy.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)

    return dilated


def filter_for_domino_mid_lines(image: numpy.ndarray) -> numpy.ndarray:
    board = grayscale(image)
    _, thresh = cv2.threshold(board, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = numpy.ones((5, 5), dtype=numpy.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    inverted = cv2.bitwise_not(dilated, dilated)

    #eroded = cv2.erode(dilated, kernel, iterations=1)
    return inverted


def draw_circles(image: numpy.ndarray, circles) -> numpy.ndarray:
    image = image.copy()
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 0, 0), -1)
    return image


def draw_contours(image: numpy.ndarray, contours: List) -> numpy.ndarray:
    image = image.copy()
    return cv2.drawContours(image, contours, -1, (0, 0, 255), 10)

def draw_lines(image: numpy.ndarray, lines: List) -> numpy.ndarray:
    image = image.copy()
    for line in lines:
        assert len(line) == 1
        x1,y1,x2,y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return image