import cv2
import numpy
from . import transforms


class TemplateMatcher:
    def __init__(self, template_image: numpy.ndarray, max_features: int):
        self.template_image: numpy.ndarray = transforms.grayscale(template_image)
        self.orb = cv2.ORB_create(max_features)

    def match(self, target_image: numpy.ndarray, match_percent: float) -> numpy.ndarray:
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
        matched = cv2.warpPerspective(target_image_color, homography, (h, w))

        return matched

    def match_all_orb(self, target_image: numpy.ndarray, match_percent: float) -> numpy.ndarray:
        target_image_color = target_image
        target_image = transforms.grayscale(target_image)

        target_keypoints, target_descriptors = self.orb.detectAndCompute(target_image, None)
        query_keypoints, query_descriptors = self.orb.detectAndCompute(self.template_image, None)

        descriptor_matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = descriptor_matcher.match(target_descriptors, query_descriptors, None)
        matches = sorted(matches, key=lambda match_point: match_point.distance)
        matches = filter(lambda match_point: match_point.distance < 95, matches)
        matches = list(matches)
        # matches = matches[:int(len(matches) * match_percent)]

        zeros = numpy.zeros((len(matches), 2), dtype=numpy.float32)
        target_points, query_points = zeros.copy(), zeros.copy()

        for i, match in enumerate(matches):
            target_points[i, :] = target_keypoints[match.queryIdx].pt
            query_points[i, :] = query_keypoints[match.trainIdx].pt

        return cv2.drawMatches(target_image_color, target_keypoints, self.template_image, query_keypoints, matches, None)

    def match_all(self, target_image: numpy.ndarray, threshold: float):
        h, w = self.template_image.shape[::]
        target_image = transforms.grayscale(target_image)

        res = cv2.matchTemplate(target_image, self.template_image, cv2.TM_CCOEFF_NORMED)
        loc = numpy.where(res >= threshold)

        matches = []
        for pt in zip(*loc[::-1]):
            matches.append([pt, (pt[0] + w, pt[1] + h)])

        print(matches)
        return matches