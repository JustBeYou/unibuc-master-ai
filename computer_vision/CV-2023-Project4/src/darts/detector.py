import logging
from typing import Union, List

import cv2
import numpy
from ultralytics import YOLO

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
        logging.info(f"Preparing board template for board type = {self.task_settings.board_type}.")

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

    def detect_arrowheads(self, image_paths: list[str]) -> List[List[str]]:
        logging.info("Loading images to memory.")
        images = [transforms.read(image_path) for image_path in image_paths]

        logging.info("Detecting darts in images using object detection model. Will take some time.")
        detection_model = self.__get_dart_detector_model()
        dart_detections = detection_model(images)

        assert len(image_paths) == len(images)
        assert len(images) == len(dart_detections)

        detections = []
        for i, (path, image, dart_detection) in enumerate(zip(image_paths, images, dart_detections)):
            logging.info(f"Extracting arrows from image {i + 1}/{len(images)}.")
            try:
                detection = self.__extract_arrowheads_from_image(image, dart_detection)
                detections.append(detection)
            except Exception as e:
                detections.append([])
                logging.error(f"Something went wrong for image: {path} - {e}. Skipping.")

                if self.debug_mode:
                    raise e

        return detections

    def __extract_arrowheads_from_image(self, image: numpy.ndarray, dart_detection) -> List[str]:
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
            contours = transforms.filter_contours(contours, min_area, max_area, 3, 15, 100, objects_in_match)
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
                intersections = transforms.line_polygon_intersection(centers[0][:2], centers[1][:2],
                                                                     box.reshape((-1, 2)))
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
                match_color = cv2.polylines(match_color, [numpy.array(triangle)], isClosed=True, color=(0, 255, 0),
                                            thickness=5)
                match_color = transforms.draw_circles(match_color, [arrowhead], custom_color=(127, 0, 255))

            match_color = transforms.draw_concentric_circles(
                match_color,
                self.task_settings.board_circle_center[0], self.task_settings.board_circle_center[1],
                self.task_settings.board_annuli,
                custom_color=(193, 182, 255)
            )

            if self.task_settings.board_type == BoardType.Classic:
                match_color = transforms.draw_sectors(
                    match_color,
                    self.task_settings.board_circle_center,
                    settings.DART_BOARD_SECTORS,
                    5,
                    offset_frac=1 / 2,
                    custom_color=(193, 182, 255)
                )

            output.debug_output_image("arrowheads", match_color)

        positions = []
        center = (self.task_settings.board_circle_center[0], self.task_settings.board_circle_center[1])

        arrowheads = [arrowhead[:2] for arrowhead in arrowheads]

        for arrowhead in arrowheads:
            if self.task_settings.board_type == BoardType.Simple:
                center_dist = transforms.dist(center, arrowhead)
                for i in range(len(self.task_settings.board_annuli) - 1):
                    a = self.task_settings.board_annuli[i]
                    b = self.task_settings.board_annuli[i + 1]

                    # print(i, a, center_dist, b)
                    if a <= center_dist <= b:
                        positions.append(10 - i)
                        break

            elif self.task_settings.board_type == BoardType.Classic:
                center_dist = transforms.dist(center, arrowhead)
                radii = self.task_settings.board_annuli

                # double bullseye
                if radii[0] <= center_dist <= radii[1]:
                    positions.append('b50')
                elif radii[1] <= center_dist <= radii[2]:
                    positions.append('b25')
                else:
                    circle_idx = None

                    for i in range(2, len(self.task_settings.board_annuli) - 1):
                        a = self.task_settings.board_annuli[i]
                        b = self.task_settings.board_annuli[i + 1]

                        if a <= center_dist <= b:
                            circle_idx = i
                            break

                    if circle_idx is None:
                        logging.error("Some arrow is outside of board?")
                        continue

                    ring = None
                    if circle_idx == 2 or circle_idx == 4:
                        ring = "s"
                    elif circle_idx == 3:
                        ring = "t"
                    elif circle_idx == 5:
                        ring = "d"

                    if ring is None:
                        logging.error("Some arrow has no ring?")
                        continue

                    sector = transforms.find_sector(arrowhead, center, settings.DART_BOARD_SECTORS, 1 / 2)
                    score = settings.DART_BOARD_SECTOR_POINTS[sector]

                    positions.append(f"{ring}{score}")
            else:
                raise Exception("Unknown board type")

        if self.task_settings.board_type == BoardType.Simple:
            positions = list(map(str, sorted(positions)))
        else:
            positions = sorted(positions)

        return positions

    @staticmethod
    def __get_dart_detector_model() -> YOLO:
        if Detector.detector_model is not None:
            return Detector.detector_model

        Detector.detector_model = YOLO(settings.YOLO_MODEL_PATH)
        return Detector.detector_model
