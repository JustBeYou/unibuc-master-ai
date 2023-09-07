import cv2
import numpy
from typing import Tuple

from . import transforms


class TemplateMatcher:
    def __init__(self, template_image: numpy.ndarray, max_features: int):
        self.template_image: numpy.ndarray = transforms.grayscale(template_image)
        self.orb = cv2.ORB_create(max_features)

    def match(self, target_image: numpy.ndarray, match_percent: float) -> Tuple[numpy.ndarray, Tuple[int, int]]:
        """
        Feature-based template matching, inspired by OpenCV documentation:
        https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        """
        target_image_color = target_image
        target_image = transforms.grayscale(target_image)

        target_keypoints, target_descriptors = self.orb.detectAndCompute(target_image, None)
        query_keypoints, query_descriptors = self.orb.detectAndCompute(self.template_image, None)

        descriptor_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = descriptor_matcher.match(target_descriptors, query_descriptors, None)
        matches = sorted(matches, key=lambda match_point: match_point.distance)
        matches = matches[:int(len(matches) * match_percent)]

        zeros = numpy.zeros((len(matches), 2), dtype=numpy.float32)
        target_points, query_points = zeros.copy(), zeros.copy()

        for i, match in enumerate(matches):
            target_points[i, :] = target_keypoints[match.queryIdx].pt
            query_points[i, :] = query_keypoints[match.trainIdx].pt

        homography, mask = cv2.findHomography(target_points, query_points, cv2.RANSAC)
        w, h = self.template_image.shape

        return homography, (h, w)