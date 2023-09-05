import enum

import cv2
import numpy
from . import transforms


class TemplateType(enum.Enum):
    SQUARE = 1
    RECTANGLE = 2
    CIRCLE = 3


def create(image: numpy.ndarray, quadrilateral: numpy.ndarray, template_type: TemplateType):
    """
        Function to crop a region of an image and create a template out of it using
        perspective transforms. Inspired by https://theailearner.com/tag/cv2-getperspectivetransform/.
    """
    if template_type == TemplateType.SQUARE:
        output_points, dimension = __create_output_square(quadrilateral)
    elif template_type == TemplateType.RECTANGLE:
        raise NotImplementedError('Rectangles are not yet supported.')
    elif template_type == TemplateType.CIRCLE:
        # to_draw = []
        # for point in quadrilateral:
        #     to_draw.append((int(point[0]), int(point[1]), 20))
        # image = transforms.draw_circles(image, circles=to_draw, custom_color=(255, 0, 0))
        output_points, dimension = __create_output_circle(quadrilateral)
    else:
        raise RuntimeError('Unknown template type passed to creator.')

    perspective_matrix = cv2.getPerspectiveTransform(quadrilateral, output_points)
    return cv2.warpPerspective(image, perspective_matrix, dimension, flags=cv2.INTER_LINEAR)


def __create_output_square(quadrilateral: numpy.ndarray) -> (numpy.ndarray, tuple[int, int]):
    length = __square_length(quadrilateral)
    return numpy.array([
        [0, 0],
        [length - 1, 0],
        [length - 1, length - 1],
        [0, length - 1]
    ], dtype=numpy.float32), (length, length)

def __create_output_circle(quadrilateral: numpy.ndarray) -> (numpy.ndarray, tuple[int, int]):
    length = __square_length(quadrilateral)
    K = length / numpy.sqrt(2)

    return numpy.array([
        [0, K - 1],
        [K - 1, 0],
        [2 * K - 1, K - 1],
        [K - 1, 2 * K - 1]
    ], dtype=numpy.float32), (int(2*K) + 500, int(2*K))

def __square_length(quadrilateral: numpy.ndarray):
    diffs = [quadrilateral[i] - quadrilateral[(i + 1) % len(quadrilateral)] for i in range(len(quadrilateral))]
    norms = map(numpy.linalg.norm, diffs)
    return int(max(norms))