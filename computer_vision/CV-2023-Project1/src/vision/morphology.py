import cv2
import numpy
from typing import Optional, List
from . import transforms, patches


def get_domino_circles(image: numpy.ndarray) -> Optional[numpy.ndarray]:
    return cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT_ALT, 1, 5,
        param1=400, param2=0.65,
        minRadius=10, maxRadius=20
    )


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

    return all_circles

def get_domino_mid_lines_from_patches(image: numpy.ndarray, patches_list: List[patches.Patch]):
    filtered_image = transforms.filter_for_domino_mid_lines(image)
    all_mid_lines = []

    for patch in patches_list:
        left_x, left_y = patch.top_left.as_tuple()
        right_x, right_y = patch.bottom_right.as_tuple()

        left_y -= PATCH_PADDING + 2
        left_x -= PATCH_PADDING + 2
        right_x += PATCH_PADDING + 2
        right_y += PATCH_PADDING + 2

        patch_image = filtered_image[left_y: right_y, left_x: right_x]
        contours = get_domino_mid_lines(patch_image)

        for contour in contours:
            contour[:, 0, 0] += left_x
            contour[:, 0, 1] += left_y
            all_mid_lines.append(contour)

    return all_mid_lines
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

        if len(approx) > 5:
            continue

        mid_lines.append(approx)

    return mid_lines

DOMINO_HALF_SIZE = 50
DOMINO_PADDING = 10
def filter_mid_lines(image: numpy.ndarray, mid_lines):
    filtered_mid_lines = []

    for mid_line in mid_lines:
        min_x = numpy.min(mid_line[:, 0, 0])
        min_y = numpy.min(mid_line[:, 0, 1])
        max_x = numpy.max(mid_line[:, 0, 0])
        max_y = numpy.max(mid_line[:, 0, 1])

        x_diff = max_x - min_x
        y_diff = max_y - min_y

        if x_diff > y_diff:
            top_left_corner = (min_x,  max(min_y - DOMINO_HALF_SIZE - DOMINO_PADDING, 0))
            top_right_corner = (max_x, min_y - DOMINO_PADDING)

            bottom_left_corner = (min_x, max_y + DOMINO_PADDING)
            bottom_right_corner = (max_x, min(max_y + DOMINO_HALF_SIZE + DOMINO_PADDING, image.shape[1]-1))
        else:
            top_left_corner = (max(min_x - DOMINO_HALF_SIZE - DOMINO_PADDING, 0), min_y)
            top_right_corner = (min_x - DOMINO_PADDING, max_y)

            bottom_left_corner = (max_x + DOMINO_PADDING, min_y)
            bottom_right_corner = (min(max_x + DOMINO_HALF_SIZE + DOMINO_PADDING, image.shape[0]-1), max_y)



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

        filtered_mid_lines.append(mid_line)

    return filtered_mid_lines
