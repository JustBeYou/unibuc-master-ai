import typing
import unittest

from darts import settings
from darts.detector import Detector


class BoardExtractionTestCase(unittest.TestCase):
    def __test_baoard_extraction(self, test_settings: settings.Settings, images: typing.List[str]):
        darts_detector = Detector(test_settings, debug_mode=True)
        darts_detector.detect_arrowheads(images[:1])

    def test_board_extraction_task_1(self):
        self.__test_baoard_extraction(settings.default_task1, settings.ALL_IMAGES_TASK_1)

    def test_board_extraction_task_2(self):
        self.__test_baoard_extraction(settings.default_task2, settings.ALL_IMAGES_TASK_2)

    def test_board_extraction_task_3(self):
        # this is not ready yet
        pass
        # for i, video in enumerate(settings.ALL_VIDEOS_TASK_3):
        #     logging.warning(f"Processing video {video}")
        #
        #     capture = cv2.VideoCapture(video)
        #     assert capture.isOpened()
        #
        #     frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #
        #     lst, fst = None, None
        #     other = []
        #     for frame_idx in range(frame_count):
        #         _, frame = capture.read()
        #         if frame_idx == 0:
        #             fst = frame
        #         elif frame_idx == frame_count - 1:
        #             lst = frame
        #
        #         if 0 < frame_idx < 5:
        #             other.append(frame)
        #     #
        #     # fg = transforms.difference(lst, fst)
        #     # fg2 = transforms.get_objects_contours(lst, fst, [], False)
        #     # fg3 = transforms.get_objects_contours(lst, fst, other, False)
        #     # fg = transforms.difference(lst, fst)
        #     fg = lst - fst
        #     fg2 = cv2.absdiff(lst, fst)
        #
        #     name = video.split('/')[-1].replace('.mp4', '')
        #     output.debug_output_image(f"Processed video last frame ({i} {name})", fg)
        #     output.debug_output_image(f"Processed video last frame ({i} {name})", fg2)
        #     # output.debug_output_image(f"Processed video last frame ({i} {name})", fg3)
        #     capture.release()


if __name__ == '__main__':
    unittest.main()
