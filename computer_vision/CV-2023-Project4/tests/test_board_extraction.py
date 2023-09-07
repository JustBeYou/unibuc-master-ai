import logging
import typing
import unittest

import cv2
import numpy

from darts import settings
from darts.detector import Detector
from darts.video_detector import detect_in_videos
from debug import output
from vision import transforms


class BoardExtractionTestCase(unittest.TestCase):
    def __test_baoard_extraction(self, test_settings: settings.Settings, images: typing.List[str]):
        darts_detector = Detector(test_settings, debug_mode=True)
        darts_detector.detect_arrowheads(images)

    def test_board_extraction_task_1(self):
        self.__test_baoard_extraction(settings.default_task1, settings.ALL_IMAGES_TASK_1)

    def test_board_extraction_task_2(self):
        self.__test_baoard_extraction(settings.default_task2, settings.ALL_IMAGES_TASK_2)

    def test_board_extraction_task_3(self):
        detect_in_videos(settings.ALL_VIDEOS_TASK_3, debug_mode=True)


if __name__ == '__main__':
    unittest.main()
