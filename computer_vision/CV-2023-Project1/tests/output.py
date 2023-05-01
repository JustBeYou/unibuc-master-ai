import os.path

import cv2
import numpy

import constants

label_counts = {}


def debug_output_image(label: str, image: numpy.ndarray):
    if not constants.WRITE_OUTPUTS:
        return

    label = '_'.join(label.lower().split(' '))
    if label not in label_counts:
        label_counts[label] = 0
        label = f"{label}.jpg"
    else:
        label_counts[label] += 1
        label = f"{label}_{label_counts[label]}.jpg"

    cv2.imwrite(os.path.join(constants.TEST_OUTPUTS_DIRECTORY, label), image)
