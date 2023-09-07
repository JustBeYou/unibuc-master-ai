import os.path

import cv2
import numpy

from darts import settings

label_counts = {}


def debug_output_image(label: str, image: numpy.ndarray):
    label = '_'.join(label.lower().split(' '))
    if label not in label_counts:
        label_counts[label] = 0
        label = f"{label}.jpg"
    else:
        label_counts[label] += 1
        label = f"{label}_{label_counts[label]}.jpg"

    cv2.imwrite(os.path.join(settings.TEST_OUTPUTS_DIRECTORY, label), image)
