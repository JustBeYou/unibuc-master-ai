import logging
from typing import List

import cv2
import numpy

from debug import output
from vision import transforms


def detect_in_videos(videos_list: List[str], debug_mode=False) -> List[str]:
    all_detections = []

    for i, video in enumerate(videos_list):
        logging.info(f"Processing video {i + 1}/{len(videos_list)}.")

        try:
            capture = cv2.VideoCapture(video)
            assert capture.isOpened()

            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            lst, fst = None, None
            for frame_idx in range(frame_count):
                _, frame = capture.read()
                if frame_idx == 0:
                    fst = frame
                elif frame_idx == frame_count - 1:
                    lst = frame

            object_mask = transforms.difference(lst, fst)

            for_circle = transforms.grayscale(fst)
            circles = transforms.get_circles(for_circle)
            circles = numpy.round(circles[0, :]).astype("int")
            # for_circle = cv2.cvtColor(for_circle, cv2.COLOR_GRAY2BGR)
            # for_circle = transforms.draw_circles(for_circle, circles, custom_color=(255, 0, 0))

            circle_zones = {
                'bottom-left': (190, 587, 107, 112),
                'top': (596, 28, 121, 108),
                'bottom': (599, 590, 109, 112)
            }
            #
            # for_circle = transforms.draw_rect(for_circle, *circle_zones['bottom-left'])
            # for_circle = transforms.draw_rect(for_circle, *circle_zones['top'])
            # for_circle = transforms.draw_rect(for_circle, *circle_zones['bottom'])

            top_count, inside_top_box = 0, False
            wider_x, taller_y = 350, 300
            for circle_x, circle_y, _ in circles:
                rect_x, rect_y, rect_w, rect_h = circle_zones['top']
                if rect_x - wider_x <= circle_x <= rect_x + rect_w + wider_x \
                        and rect_y <= circle_y <= rect_y + rect_h + taller_y:
                    top_count += 1

                if rect_x <= circle_x <= rect_x + rect_w and rect_y <= circle_y <= rect_y + rect_h:
                    inside_top_box += 1

            bottom_left_count, inside_bottom_left_box = 0, 0
            bigger = 250
            for circle_x, circle_y, _ in circles:
                rect_x, rect_y, rect_w, rect_h = circle_zones['bottom-left']
                if rect_x - bigger <= circle_x <= rect_x + rect_w + bigger \
                        and rect_y - bigger <= circle_y <= rect_y + rect_h + bigger:
                    bottom_left_count += 1

                if rect_x <= circle_x <= rect_x + rect_w and rect_y <= circle_y <= rect_y + rect_h:
                    inside_bottom_left_box += 1

            if top_count == 1 and inside_top_box == 1:
                zone = 'top'
                lst = transforms.draw_rect(lst, *circle_zones['top'], color=(0, 0, 255))
            elif bottom_left_count == 1 and inside_bottom_left_box == 1:
                zone = 'bottom-left'
                object_mask[:, 580:] = 0
                lst = transforms.draw_rect(lst, *circle_zones['bottom-left'], color=(0, 0, 255))
            else:
                zone = 'bottom'
                lst = transforms.draw_rect(lst, *circle_zones['bottom'], color=(0, 0, 255))

            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = transforms.filter_contour_in_video(contours, 100, 50_000)

            points_in_contours = contours[0].reshape(-1, 2)
            triangle_x, triangle_y = 999999999, -1000000000
            for (x, y) in points_in_contours:
                triangle_x = min(triangle_x, x)
                triangle_y = max(triangle_y, y)
            near_triangle_arrow_corner = (triangle_x, triangle_y)

            all_points = numpy.vstack(contours)
            _, triangle = cv2.minEnclosingTriangle(all_points)
            triangle = triangle.reshape((-1, 2)).astype(int)
            _, triangle_arrow_corner = transforms.shortest_side_info(triangle)

            # outframe = object_mask
            # outframe = cv2.cvtColor(outframe, cv2.COLOR_GRAY2BGR)
            outframe = lst

            if debug_mode:
                outframe = cv2.drawContours(outframe, contours, -1, (0, 255, 0), 2)
                outframe = cv2.polylines(outframe, [numpy.array(triangle)], isClosed=True, color=(255, 0, 0), thickness=5)
                outframe = transforms.draw_circles(outframe, [(*triangle_arrow_corner, 3)], custom_color=(127, 0, 255))
                outframe = transforms.draw_circles(outframe, [(*near_triangle_arrow_corner, 3)], custom_color=(150, 0, 255))

            centers = {
                'bottom-left': (243, 644),
                'top': (657, 78),
                'bottom': (658, 654),
            }

            sectors = {
                'bottom-left': [118, 101, 84, 65, 46],
                'top': ([170, 189, 208, 227, 246, 264, 281, 298, 315, 332, 349, 8]),
                'bottom': [188, 169, 151, 134, 118, 102, 85, 67, 49, 27],
            }

            distances = {
                'bottom-left': [0, 20, 45, 276, 304, 455, 5000],
                'top': [0, 20, 48, 283, 313, 464, 5000],
                'bottom': [0, 20, 48, 283, 313, 464, 5000],
            }

            scores = {
                'bottom-left': [5, 20, 1, 18],
                'top': ([11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 9]),
                'bottom': [11, 14, 9, 12, 5, 20, 1, 18, 4],
            }

            defaults = {
                'bottom-left': 's1',
                'top': 's3',
                'bottom': 's5',
            }

            result = defaults[zone]
            ring, score = None, None

            d = transforms.dist(triangle_arrow_corner, centers[zone])
            if d <= distances[zone][1]:
                ring, score = 'b', '50'
            elif d <= distances[zone][2]:
                ring, score = 'b', '25'
            else:
                ring_index = None
                for i in range(2, len(distances[zone]) - 1):
                    a = distances[zone][i]
                    b = distances[zone][i + 1]

                    if a <= d <= b:
                        ring_index = i
                        break

                if ring_index == 2 or ring_index == 4:
                    ring = "s"
                elif ring_index == 3:
                    ring = "t"
                elif ring_index == 5:
                    ring = "d"

                angle = transforms.find_angle(triangle_arrow_corner, centers[zone])

                score_idx = -1
                for i in range(len(sectors[zone]) - 2):
                    a = sectors[zone][i]
                    b = sectors[zone][i + 1]

                    if min(a, b) <= angle <= max(a, b):
                        score_idx = i
                        break

                score = scores[zone][score_idx]

            if ring is not None and score is not None:
                result = f"{ring}{score}"

            capture.release()
            all_detections.append(result)

            if debug_mode:
                outframe = transforms.draw_specific_sectors(outframe, centers[zone], sectors[zone], 1)
                name = video.split('/')[-1].replace('.mp4', '')
                output.debug_output_image(f"Circle detection ({i} {name})", outframe)

        except Exception as e:
            logging.error(f"Something went wrong for video: {video} - {e}. Skipping.")
            all_detections.append('s1')

    return all_detections