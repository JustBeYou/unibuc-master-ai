from typing import List

import cv2
import numpy
from skimage.metrics import structural_similarity

def draw_circles(image: numpy.ndarray, circles, custom_color=(0, 0, 0)) -> numpy.ndarray:
    image = image.copy()
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, custom_color, -1)
    return image


def draw_concentric_circles(image: numpy.ndarray, center_x, center_y, radii, custom_color=(0, 0, 0)) -> numpy.ndarray:
    image = image.copy()
    for r in radii:
        cv2.circle(image, (center_x, center_y), r, custom_color, thickness=5)
    return image


def draw_sectors(image, center, num_sectors, line_thickness, offset_frac=0, custom_color=(255, 255, 255)):
    height, width = image.shape[:2]
    angle_step = 360 / num_sectors
    max_len = max(width, height) * 2

    for i in range(num_sectors):
        angle = numpy.radians(offset_frac * angle_step + i * angle_step)
        end_x = int(center[0] + max_len * numpy.cos(angle))
        end_y = int(center[1] + max_len * numpy.sin(angle))
        cv2.line(image, center, (end_x, end_y), custom_color, line_thickness)

    return image


def thresh_invert(image: numpy.ndarray) -> numpy.ndarray:
    _, thresh = cv2.threshold(grayscale(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(thresh, thresh)

def better_contours(image: numpy.ndarray) -> numpy.ndarray:
    image = cv2.medianBlur(image, 5)
    kernel = numpy.ones((5, 5), dtype=numpy.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # kernel = numpy.ones((3, 3), dtype=numpy.uint8)
    # dilate = cv2.dilate(blur, kernel)
    #
    # kernel = numpy.ones((3, 3), dtype=numpy.uint8)
    # erode = cv2.erode(dilate, kernel)
    return image

def invert(img: numpy.ndarray) -> numpy.ndarray:
    return cv2.bitwise_not(img)

def predif(img):
    return some_filters(img)

def postdir(img):
    return img

def difference(modified: numpy.ndarray, original: numpy.ndarray) -> numpy.ndarray:
    # original = align_images(original, modified)
    # print(original.shape, modified.shape)
    # (score, diff) = structural_similarity(grayscale(original), grayscale(modified), full=True)
    return postdir(predif(modified) - predif(original))

def crop(img: numpy.ndarray, top_left, bottom_right):
    return img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

def read(path: str) -> numpy.ndarray:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def grayscale(image: numpy.ndarray) -> numpy.ndarray:
    if len(image.shape) > 2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def align_images(A, B):
    A_gray = grayscale(A)
    B_gray = grayscale(B)

    result = cv2.matchTemplate(A_gray, B_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    x_shift, y_shift = max_loc

    M = numpy.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(A, M, B.shape[1::-1])

def get_objs_contours(img, tmpl):
    backSub = cv2.createBackgroundSubtractorMOG2()
    tmpl = align_images(tmpl, img)
    backSub.apply(grayscale(tmpl))
    x = backSub.apply(grayscale(img))
    return some_filters(x)
    # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return thresh

def enclosing_rectangle(polygon):
    if len(polygon) != 4:
        raise ValueError("Polygon should have 4 points.")

    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    return ((min_x, min_y), (max_x, max_y))

def expand_image_right(img, w):
    h, width, _ = img.shape
    expanded_img = numpy.zeros((h, width + w, 3), dtype=img.dtype)
    expanded_img[:, :width] = img
    return expanded_img

def some_filters(image):
    board = grayscale(image)
    blur = cv2.GaussianBlur(board, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # inverted = cv2.bitwise_not(thresh, thresh)
    kernel = numpy.ones((3, 3), dtype=numpy.uint8)
    x = cv2.erode(thresh, kernel, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    return cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)

def filter_contours(contours, min_area, max_area, min_sides, max_sides, min_width, polygon):
    contours = [contour for contour in contours if min_area < cv2.contourArea(contour) < max_area]

    output = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        min_x = numpy.min(approx[:, 0, 0])
        min_y = numpy.min(approx[:, 0, 1])
        max_x = numpy.max(approx[:, 0, 0])
        max_y = numpy.max(approx[:, 0, 1])

        x_diff = max_x - min_x
        y_diff = max_y - min_y

        if x_diff < min_width:
            continue

        if len(approx) < min_sides or len(approx) > max_sides:
            continue

        # is_inside = True
        # for point in contour:
        #     if cv2.pointPolygonTest(polygon, tuple(point.astype(numpy.float32)[0]), False) < 0:  # Point is outside
        #         is_inside = False
        #         break
        # if not is_inside:
        #     continue

        output.append(contour)

    return sorted(output, key=cv2.contourArea, reverse=True)[:2]


def line_polygon_intersection(p1, p2, polygon):
    """Return intersection points between line and polygon."""
    intersections = []
    bbox = (
    min(p[0] for p in polygon), min(p[1] for p in polygon), max(p[0] for p in polygon), max(p[1] for p in polygon))

    for i in range(len(polygon)):
        p3 = polygon[i]
        p4 = polygon[(i + 1) % len(polygon)]

        intr = intersect(p3, p4, p1, p2)
        if intr is not None:
            intersections.append(intr)

    return intersections


def line_parameters(p1, p2):
    """Returns the coefficients a, b, c of the line ax + by = c passing through points p1 and p2."""
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = A * p1[0] + B * p1[1]
    return A, B, C


def intersect(A, B, C, D):
    """Returns the intersection point (if it exists) of segment AB with the line on which CD lies."""
    A1, B1, C1 = line_parameters(A, B)
    A2, B2, C2 = line_parameters(C, D)

    determinant = A1 * B2 - A2 * B1

    # If determinant is zero, the lines are parallel and won't intersect
    if determinant == 0:
        return None

    x = (C1 * B2 - C2 * B1) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    # Check if (x, y) is on segment AB
    if (min(A[0], B[0]) <= x <= max(A[0], B[0])) and (min(A[1], B[1]) <= y <= max(A[1], B[1])):
        return (int(x), int(y))

    return None


def shortest_side_info(triangle):
    # Calculate pairwise distances
    distances = [
        (numpy.linalg.norm(numpy.array(triangle[0]) - numpy.array(triangle[1])), (triangle[0], triangle[1], triangle[2])),
        (numpy.linalg.norm(numpy.array(triangle[1]) - numpy.array(triangle[2])), (triangle[1], triangle[2], triangle[0])),
        (numpy.linalg.norm(numpy.array(triangle[0]) - numpy.array(triangle[2])), (triangle[0], triangle[2], triangle[1]))
    ]

    # Sort the distances and select the shortest one
    shortest = sorted(distances, key=lambda x: x[0])[0]

    # Calculate the middle point
    middle_point = tuple((numpy.array(shortest[1][0]) + numpy.array(shortest[1][1])) / 2)

    return middle_point, shortest[1][2]