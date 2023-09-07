import cv2
import numpy
from ultralytics import YOLO
from typing import Union

from darts import settings
from darts.settings import BoardType
from debug import output
from vision import transforms, templates, template_matcher


class Detector:
    detector_model: Union[YOLO, None] = None

    def __init__(self, task_settings: settings.Settings, debug_mode: bool = False):
        self.task_settings: settings.Settings = task_settings
        self.__init_template_matcher()
        self.debug_mode = debug_mode

    def __init_template_matcher(self):
        board_for_template = transforms.read(self.task_settings.board_for_template_path)
        template_image = templates.create(
            board_for_template,
            self.task_settings.board_template_quadrilateral,
            templates.TemplateType.CIRCLE
        )

        self.matcher = template_matcher.TemplateMatcher(
            template_image,
            self.task_settings.board_match_max_features
        )

    def detect_arrowheads(self, image_paths: list[str]):
        images = [transforms.read(image_path) for image_path in image_paths]
        detection_model = self.__get_dart_detector_model()
        dart_detections = detection_model(images)
        assert len(images) == len(dart_detections)

        return [
            self.__extract_arrowheads_from_image(image, dart_detection)
            for image, dart_detection in zip(images, dart_detections)
        ]

    def __extract_arrowheads_from_image(self, image: numpy.ndarray, dart_detection):
        homography, (h, w) = self.matcher.match(image, self.task_settings.board_match_percent)
        match_color = cv2.warpPerspective(image, homography, (h, w))

        boxes, rects, triangles, all_contours = [], [], [], []

        objects_in_match = transforms.get_objects_contours(match_color, self.matcher.template_image)

        for box, conf in zip(dart_detection.boxes.xyxy, dart_detection.boxes.conf.reshape(-1)):
            if conf < 0.6:
                continue

            box = [
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
            ]
            box = numpy.array(box, dtype=numpy.float64).reshape((-1, 1, 2))
            box = cv2.perspectiveTransform(box, homography.astype(numpy.float64))
            box = box.astype(numpy.int32)
            boxes.append(box)

            rect = transforms.enclosing_rectangle(box.reshape((-1, 2)))
            rects.append(rect)

            rect_only = objects_in_match[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
            contours, _ = cv2.findContours(rect_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 3_000
            max_area = 100_000
            contours = transforms.filter_contours(contours, min_area, max_area, 3, 15, 100, box)
            contours = [numpy.add(contour, [rect[0][0], rect[0][1]]) for contour in contours]
            all_contours.append(contours)

            all_points = numpy.vstack(contours)
            retval, triangle = cv2.minEnclosingTriangle(all_points)
            triangles.append(triangle.reshape((-1, 2)).astype(int))

        arrowheads = []

        for contours, box, triangle in zip(all_contours, boxes, triangles):
            centers = []
            for contour in contours:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY, 10])

            if len(centers) == 2:
                intersections = transforms.line_polygon_intersection(centers[0][:2], centers[1][:2], box.reshape((-1, 2)))
                intersections = [(x, y, 15) for x, y in intersections]
            else:
                mid_point, corner = transforms.shortest_side_info(triangle)
                intersections = transforms.line_polygon_intersection(mid_point, corner, box.reshape((-1, 2)))
                intersections = [(x, y, 15) for x, y in intersections]

            assert len(intersections) > 0

            the_point = min(intersections, key=lambda inter: inter[0])
            arrowheads.append(the_point)

            if self.debug_mode:
                match_color = transforms.draw_circles(match_color, centers, custom_color=(0, 0, 255))
                match_color = cv2.drawContours(match_color, contours, -1, (0, 255, 0), 2)

        if self.debug_mode:
            for box, rect, triangle, arrowhead in zip(boxes, rects, triangles, arrowheads):
                match_color = cv2.polylines(match_color, [box], True, (255, 0, 0), 5)
                match_color = cv2.rectangle(match_color, rect[0], rect[1], (193, 182, 255), 5)
                match_color = cv2.polylines(match_color, [numpy.array(triangle)], isClosed=True, color=(0, 255, 0), thickness=5)
                match_color = transforms.draw_circles(match_color, [arrowhead], custom_color=(127, 0, 255))

            if self.task_settings.board_type == BoardType.Simple:
                match_color = transforms.draw_concentric_circles(
                    match_color,
                    self.task_settings.board_circle_center[0], self.task_settings.board_circle_center[1],
                    self.task_settings.board_annuli,
                    custom_color=(193,182,255)
                )
            elif self.task_settings.board_type == BoardType.Classic:
                match_color = transforms.draw_sectors(
                    match_color,
                    self.task_settings.board_circle_center,
                    settings.DART_BOARD_SECTORS,
                    5,
                    offset_frac=1/2,
                    custom_color=(193, 182, 255)
                )

            output.debug_output_image("arrowheads", match_color)

    @staticmethod
    def __get_dart_detector_model() -> YOLO:
        if Detector.detector_model is not None:
            return Detector.detector_model

        Detector.detector_model = YOLO(settings.YOLO_MODEL_PATH)
        return Detector.detector_model
