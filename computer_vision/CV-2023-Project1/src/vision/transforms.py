import cv2
import numpy
import numpy as np

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


def draw_patches(image: numpy.ndarray, patches_list: List[patches.Patch]) -> numpy.ndarray:
    # image = black_and_white(grayscale(image))
    for patch in patches_list:
        left_x, left_y = patch.top_left.as_tuple()
        right_x, right_y = patch.bottom_right.as_tuple()

        # mean_blue = numpy.mean(image[left_y: right_y, left_x: right_x, 0])
        # mean_green = numpy.mean(image[left_y: right_y, left_x: right_x, 1])
        # mean_red = numpy.mean(image[left_y: right_y, left_x: right_x, 2])

        # image[left_y: right_y, left_x: right_x] = (mean_blue, mean_green, mean_red)

        # image[left_y: right_y, left_x: right_x] = numpy.mean(image[left_y: right_y, left_x: right_x])

        # image[left_y: right_y, left_x: right_x] = dominant_color(image[left_y: right_y, left_x: right_x])

        image = cv2.rectangle(
            image,
            patch.top_left.as_tuple(),
            patch.bottom_right.as_tuple(),
            (255, 255, 255),
            patch.line_thickness
        )
    return image


def blur_and_thresh(image: numpy.ndarray) -> numpy.ndarray:
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
    # blurred = cv2.GaussianBlur(image, (5, 5), 3)
    # return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def dominant_color(image: numpy.ndarray) -> numpy.ndarray:
    pixels = numpy.float32(image.reshape(-1, 3))

    n_colors = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = numpy.unique(labels, return_counts=True)

    return palette[numpy.argmax(counts)]

def black_and_white(image: numpy.ndarray) -> numpy.ndarray:
    board = grayscale(image)
    blur = cv2.GaussianBlur(board, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh, thresh)

    kernel = numpy.ones((5, 5), dtype=numpy.uint8)

    return cv2.dilate(inverted, kernel, iterations=1)

def get_circles(image: numpy.ndarray):
    bw = black_and_white(image)
    output = bw.copy()
    circles = cv2.HoughCircles(output, cv2.HOUGH_GRADIENT_ALT, 1, 10,
                              param1=200, param2=0.90,
                              minRadius=10,
                              maxRadius=105
                              )

    return circles

def draw_circles(image: numpy.ndarray, circles) -> numpy.ndarray:
    output = image.copy()
    circles = numpy.round(circles[0, :]).astype("int")
    print(len(circles))
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 0, 255), -1)
    return output

def draw_bullet_matches(image, matches):
    image = image.copy()
    for bullet_match in matches:
        cv2.rectangle(image, *bullet_match, (0, 0, 255), 2)
    return image