import enum

import cv2
import numpy


class TemplateType(enum.Enum):
    SQUARE = 1
    RECTANGLE = 2


def create(image: numpy.ndarray, quadrilateral: numpy.ndarray, template_type: TemplateType):
    """
        Function to crop a region of an image and create a template out of it using
        perspective transforms. Inspired by https://theailearner.com/tag/cv2-getperspectivetransform/.
    """
    if template_type == TemplateType.SQUARE:
        output_points, dimension = __create_output_square(quadrilateral)
    elif template_type == TemplateType.RECTANGLE:
        raise NotImplementedError('Rectangles are not yet supported.')
    else:
        raise RuntimeError('Unknown template type passed to creator.')

    perspective_matrix = cv2.getPerspectiveTransform(quadrilateral, output_points)
    return cv2.warpPerspective(image, perspective_matrix, dimension, flags=cv2.INTER_LINEAR)


def __create_output_square(quadrilateral: numpy.ndarray) -> (numpy.ndarray, tuple[int, int]):
    diffs = [quadrilateral[i] - quadrilateral[(i + 1) % len(quadrilateral)] for i in range(len(quadrilateral))]
    norms = map(numpy.linalg.norm, diffs)
    length = int(max(norms))

    return numpy.array([
        [0, 0],
        [length - 1, 0],
        [length - 1, length - 1],
        [0, length - 1]
    ], dtype=numpy.float32), (length, length)
