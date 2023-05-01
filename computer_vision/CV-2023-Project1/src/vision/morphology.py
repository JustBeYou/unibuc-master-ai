import cv2
import imutils
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

def get_domino_mid_lines(image: numpy.ndarray) -> List:
    """
    Returns the n-th polygon with m sides from a given image. Inspired by
    https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/.
    """
    mid_lines = []

    filtered_image = transforms.filter_for_domino_mid_lines(image)
    contours = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10_000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        #
        # if len(approx) != 4:
        #     continue

        mid_lines.append(approx)

    return mid_lines
