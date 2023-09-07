import logging
import time
import typing
import unittest

import cv2
import numpy

import constants
import output
from darts import settings
from vision import templates, transforms, template_matcher

from ultralytics import YOLO

model = None

def lazy_load_model():
    global model
    if model is not None:
        return model

    model = YOLO(f'{settings.DATA_DIRECTORY}/best.pt')
    return model

class BoardExtractionTestCase(unittest.TestCase):
    def __test_baoard_extraction(self, test_settings: settings.Settings, images: typing.List[str]):
        model = lazy_load_model()

        board_for_template = transforms.read(test_settings.board_for_template_path)
        template_image = templates.create(
            board_for_template,
            test_settings.board_template_quadrilateral,
            templates.TemplateType.CIRCLE
        )
        output.debug_output_image("Template for board extraction", template_image)

        matcher = template_matcher.TemplateMatcher(
            template_image,
            test_settings.board_match_max_features
        )

        start_time = time.time()

        images = images
        results = model(images, verbose=True)


        for i, image_path in enumerate(images):
            print(i, image_path)
            image = transforms.read(image_path)
            homography, (h, w) = matcher.match(image, test_settings.board_match_percent)

            match_color = cv2.warpPerspective(image, homography, (h, w))

            boxes = []
            rects = []
            triangles = []
            all_contours = []

            name = image_path.split('/')[-1].replace('.jpg', '')
            # output.debug_output_image(f"Board original ({i} {name})", match_color)
            match_gray = transforms.get_objs_contours(match_color, template_image)

            for box, conf in zip(results[i].boxes.xyxy, results[i].boxes.conf.reshape(-1)):
                if conf < 0.6:
                    continue

                box = [
                    [box[0], box[1]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                    [box[0], box[3]],
                ]
                box = numpy.array(box, dtype=numpy.float64).reshape(-1, 1, 2)
                box = cv2.perspectiveTransform(box, homography.astype(numpy.float64))
                box = box.astype(numpy.int32)
                boxes.append(box)

                rect = transforms.enclosing_rectangle(box.reshape(-1, 2))
                rects.append(rect)

                rect_only = match_gray[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
                contours, _ = cv2.findContours(rect_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = 3_000
                max_area = 100_000
                contours = transforms.filter_contours(contours, min_area, max_area, 3, 15, 100, box)
                print(list(map(cv2.contourArea, contours)))
                contours = [numpy.add(contour, [rect[0][0], rect[0][1]]) for contour in contours]
                all_contours.append(contours)

                all_points = numpy.vstack(contours)
                retval, triangle = cv2.minEnclosingTriangle(all_points)
                triangles.append(triangle.reshape(-1, 2).astype(int))

            # match_color = cv2.cvtColor(match_color, cv2.COLOR_GRAY2BGR)


            for contours, box, triangle in zip(all_contours, boxes, triangles):
                centers = []
                for contour in contours:
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append([cX, cY, 10])

                if len(centers) == 2:
                    intersections = transforms.line_polygon_intersection(centers[0][:2], centers[1][:2], box.reshape(-1, 2))
                    intersections = [(x, y, 15) for x, y in intersections]
                else:
                    mid_point, corner = transforms.shortest_side_info(triangle)
                    intersections = transforms.line_polygon_intersection(mid_point, corner, box.reshape(-1, 2))
                    intersections = [(x, y, 15) for x, y in intersections]

                assert len(intersections) > 0
                the_point = min(intersections, key=lambda inter: inter[0])

                match_color = transforms.draw_circles(match_color, [the_point], custom_color=(127, 0, 255))

                # match_color = transforms.draw_circles(match_color, centers, custom_color=(0, 0, 255))
                # match_color = cv2.drawContours(match_color, contours, -1, (0, 255, 0), 2)

            # for box, rect, triangle in zip(boxes, rects, triangles):
                # match_color = cv2.polylines(match_color, [box], True, (255, 0, 0), 5)
                # match_color = cv2.rectangle(match_color, rect[0], rect[1], (193, 182, 255), 5)
                # match_color = cv2.polylines(match_color, [numpy.array(triangle)], isClosed=True, color=(0, 255, 0), thickness=5)


            # match = transforms.grayscale(match_color)
            # self.assertEqual(match.shape, template_image.shape)
            name = image_path.split('/')[-1].replace('.jpg', '')
            print(i, name)

            # match_color = transforms.draw_concentric_circles(
            #     match_color, test_settings.board_circle_center[0], test_settings.board_circle_center[1],
            #     test_settings.board_annuli,
            #     custom_color=(193,182,255)
            # )
            # if 'task2' in image_path.lower():
            #     match_color = transforms.draw_sectors(
            #         match_color,
            #         test_settings.board_circle_center,
            #         settings.DART_BOARD_SECTORS,
            #         5,
            #         offset_frac=1/2,
            #         custom_color=(193, 182, 255)
            #     )


            # match_color = transforms.some_filters(match_color)
            output.debug_output_image(f"Processed board ({i} {name})", match_color)
            logging.debug(
                f"Time elapsed {int(time.time() - start_time)}s ({i + 1}/{len(images)})")


        # for (i, image_path), result in zip(enumerate(images), results):
        #     name = image_path.split('/')[-1].replace('.jpg', '')
        #     print(i, name)
        #     img = transforms.read(image_path)
        #     homography, (h, w) = matcher.match(img, test_settings.board_match_percent)
        #     # img = result.plot(kpt_radius=10)
        #     img = cv2.warpPerspective(img, homography, (h, w))
        #
        #     output.debug_output_image(f"yolo board ({i} {name})", img)

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
        #     # fg2 = transforms.get_objs_contours(lst, fst, [], False)
        #     # fg3 = transforms.get_objs_contours(lst, fst, other, False)
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
