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
    black_and_white_image = transforms.black_and_white(image)
    all_circles = []

    for patch in patches_list:
        left_x, left_y = patch.top_left.as_tuple()
        right_x, right_y = patch.bottom_right.as_tuple()

        left_y -= PATCH_PADDING
        left_x -= PATCH_PADDING
        right_x += PATCH_PADDING
        right_y += PATCH_PADDING

        patch_image = black_and_white_image[left_y: right_y, left_x: right_x]
        circles = get_domino_circles(patch_image)
        if circles is None:
            continue

        circles = numpy.round(circles[0, :]).astype("int")
        circles[:, 0] += left_x
        circles[:, 1] += left_y

        all_circles.extend(circles)

    return all_circles
