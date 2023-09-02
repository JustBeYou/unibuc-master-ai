import dataclasses
import typing
from typing import Optional, List

import cv2
import numpy

from . import transforms, patches


def get_domino_circles(image: numpy.ndarray) -> Optional[numpy.ndarray]:
    return cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT_ALT, 1, 5,
        param1=400, param2=0.65,
        minRadius=10, maxRadius=20
    )


def should_merge_circles(x1, y1, r1, x2, y2, r2):
    d = numpy.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if d <= r1 - r2:
        return True
    elif d <= r2 - r1:
        return True
    elif d < r1 + r2 and r1 + r2 - d > 10:
        return True
    elif d == r1 + r2:
        return False
    else:
        return False


def remove_overlapping_circles(circles):
    for i in range(len(circles)):
        for j in range(len(circles)):
            if i == j: continue
            if circles[i] is None: continue
            if circles[j] is None: continue

            if should_merge_circles(*circles[i], *circles[j]):
                circles[j] = None

    return [circle for circle in circles if circle is not None]


PATCH_PADDING = 12


def get_domino_circles_from_patches(image: numpy.ndarray, patches_list: List[patches.Patch]) -> List:
    filtered_image = transforms.filter_for_domino_circles(image)
    all_circles = []

    for patch in patches_list:
        left_x, left_y = patch.top_left.as_tuple()
        right_x, right_y = patch.bottom_right.as_tuple()

        left_y -= PATCH_PADDING
        left_x -= PATCH_PADDING
        right_x += PATCH_PADDING
        right_y += PATCH_PADDING

        patch_image = filtered_image[left_y: right_y, left_x: right_x]
        circles = get_domino_circles(patch_image)
        if circles is None:
            continue

        circles = numpy.round(circles[0, :]).astype("int")
        circles[:, 0] += left_x
        circles[:, 1] += left_y

        all_circles.extend(circles)

    return remove_overlapping_circles(all_circles)


def get_domino_mid_lines(image: numpy.ndarray) -> List:
    """
    Returns the n-th polygon with m sides from a given image. Inspired by
    https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/.
    """
    mid_lines = []

    image = transforms.filter_for_domino_mid_lines(image)
    contours = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300 or area > 2000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        min_x = numpy.min(approx[:, 0, 0])
        min_y = numpy.min(approx[:, 0, 1])
        max_x = numpy.max(approx[:, 0, 0])
        max_y = numpy.max(approx[:, 0, 1])

        x_diff = max_x - min_x
        y_diff = max_y - min_y

        max_diff = max(x_diff, y_diff)
        min_diff = min(x_diff, y_diff)

        ratio = max_diff / min_diff

        if ratio < 3:
            continue

        if len(approx) > 6:
            continue

        mid_lines.append(approx)

    return mid_lines


@dataclasses.dataclass
class MidLine:
    contour: typing.Any
    min_x: int
    max_x: int
    min_y: int
    max_y: int


DOMINO_HALF_SIZE = 50
DOMINO_PADDING = 10


def filter_mid_lines(image: numpy.ndarray, mid_lines) -> List[MidLine]:
    filtered_mid_lines = []

    for mid_line in mid_lines:
        min_x = numpy.min(mid_line[:, 0, 0])
        min_y = numpy.min(mid_line[:, 0, 1])
        max_x = numpy.max(mid_line[:, 0, 0])
        max_y = numpy.max(mid_line[:, 0, 1])

        x_diff = max_x - min_x
        y_diff = max_y - min_y

        if x_diff > y_diff:
            top_left_corner = (min_x, max(min_y - DOMINO_HALF_SIZE - DOMINO_PADDING, 0))
            top_right_corner = (max_x, min_y - DOMINO_PADDING)

            bottom_left_corner = (min_x, max_y + DOMINO_PADDING)
            bottom_right_corner = (max_x, min(max_y + DOMINO_HALF_SIZE + DOMINO_PADDING, image.shape[1] - 1))
        else:
            top_left_corner = (max(min_x - DOMINO_HALF_SIZE - DOMINO_PADDING, 0), min_y)
            top_right_corner = (min_x - DOMINO_PADDING, max_y)

            bottom_left_corner = (max_x + DOMINO_PADDING, min_y)
            bottom_right_corner = (min(max_x + DOMINO_HALF_SIZE + DOMINO_PADDING, image.shape[0] - 1), max_y)

        top_region = image[
                     top_left_corner[1]:top_right_corner[1],
                     top_left_corner[0]:top_right_corner[0]
                     ]
        if top_region.shape[0] == 0 or top_region.shape[1] == 0:
            continue

        top_mean = numpy.mean(top_region)
        # cv2.rectangle(image, top_left_corner, top_right_corner, (128, 128, 128), -1)

        if top_mean == numpy.nan:
            continue

        bottom_region = image[
                        bottom_left_corner[1]:bottom_right_corner[1],
                        bottom_left_corner[0]:bottom_right_corner[0]
                        ]

        if bottom_region.shape[0] == 0 or bottom_region.shape[1] == 0:
            continue

        bottom_mean = numpy.mean(bottom_region)
        # cv2.rectangle(image, bottom_left_corner, bottom_right_corner, (128, 128, 128), -1)

        if bottom_mean == numpy.nan:
            continue

        if top_mean > 10 or bottom_mean > 10:
            continue

        filtered_mid_lines.append(MidLine(mid_line, min_x, max_x, min_y, max_y))

    return filtered_mid_lines


LinesMatrix = typing.List[typing.List[typing.Optional[MidLine]]]


def lines_to_grid(
        lines: List[MidLine],
        patches: List[List[patches.Patch]],
        patch_x_size: int,
        patch_y_size: int,
        margin_size: int,
        columns: int,
        rows: int,
):
    lines = sorted(lines, key=lambda elem: (elem.min_x, elem.min_y))

    vertical_grid: LinesMatrix = [[None for _ in range(columns)] for _ in range(rows)]
    horizontal_grid: LinesMatrix = [[None for _ in range(columns)] for _ in range(rows)]

    for line in lines:
        x_diff = line.max_x - line.min_x
        y_diff = line.max_y - line.min_y

        y_mean = line.min_y + y_diff / 2
        x_mean = line.min_x + x_diff / 2

        column = int((x_mean - margin_size) / patch_x_size)
        row = int((y_mean - margin_size) / patch_y_size)

        if row >= rows:
            row = rows - 1
        elif column >= columns:
            column = columns - 1

        current_patch = patches[row][column]

        if x_diff > y_diff:
            # horizontal
            y_min = current_patch.top_left.y
            y_max = current_patch.bottom_right.y

            if numpy.abs(y_min - y_mean) < numpy.abs(y_max - y_mean):
                row -= 1

            horizontal_grid[row][column] = line
            # print('horizontal', row, column)
        else:
            # vertical
            x_min = current_patch.top_left.x
            x_max = current_patch.bottom_right.x

            if numpy.abs(x_min - x_mean) < numpy.abs(x_max - x_mean):
                column -= 1

            vertical_grid[row][column] = line
            # print('vertical', row, column)

        # print(x_mean, y_mean, margin_size, patch_x_size, patch_y_size)

    # print("###################")

    for column in range(columns):
        for row in range(rows - 1):
            if horizontal_grid[row][column] and horizontal_grid[row + 1][column]:
                horizontal_grid[row + 1][column] = None

            # if horizontal_grid[row][column]:
            #     print('horizontal', row, column)

    for row in range(rows):
        for column in range(columns - 1):
            if vertical_grid[row][column] and vertical_grid[row][column + 1]:
                vertical_grid[row][column + 1] = None

            # if vertical_grid[row][column]:
            #     print('vertical', row, column)

    all_lines = [line for row in vertical_grid for line in row if line is not None] + \
                [line for row in horizontal_grid for line in row if line is not None]

    # print('========================')
    return all_lines, horizontal_grid, vertical_grid
