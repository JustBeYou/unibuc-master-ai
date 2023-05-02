import cv2
import numpy


DOMINO_HALF_SIZE = 115
DOMINO_ADDITIONAL_WIDTH = 1
DOMINO_PADDING = 3

def create_board_from(horizontal_grid, vertical_grid, circles, rows, columns, image):
    board = [[-1 for _ in range(columns)] for _ in range(rows)]
    directions = [['' for _ in range(columns)] for _ in range(rows)]

    circles = circles.copy()

    for row in range(rows):
        for column in range(columns):
            is_horizontal = False

            if horizontal_grid[row][column] is not None:
                is_horizontal = True
                mid_line = horizontal_grid[row][column]

                min_x = mid_line.min_x
                min_y = mid_line.min_y
                max_x = mid_line.max_x
                max_y = mid_line.max_y

                top_left_corner = (max(min_x - DOMINO_ADDITIONAL_WIDTH, 0), max(min_y - DOMINO_HALF_SIZE - DOMINO_PADDING, 0))
                top_right_corner = (min(max_x + DOMINO_ADDITIONAL_WIDTH, image.shape[0] - 1), min_y - DOMINO_PADDING)

                bottom_left_corner = (max(min_x - DOMINO_ADDITIONAL_WIDTH, 0), max_y + DOMINO_PADDING)
                bottom_right_corner = (min(max_x + DOMINO_ADDITIONAL_WIDTH, image.shape[0] - 1), min(max_y + DOMINO_HALF_SIZE + DOMINO_PADDING, image.shape[1] - 1))
            elif vertical_grid[row][column] is not None:

                mid_line = vertical_grid[row][column]

                min_x = mid_line.min_x
                min_y = mid_line.min_y
                max_x = mid_line.max_x
                max_y = mid_line.max_y

                top_left_corner = (max(min_x - DOMINO_HALF_SIZE - DOMINO_PADDING, 0), max(min_y - DOMINO_ADDITIONAL_WIDTH, 0))
                top_right_corner = (min_x - DOMINO_PADDING, min(max_y + DOMINO_ADDITIONAL_WIDTH, image.shape[1] -1))

                bottom_left_corner = (max_x + DOMINO_PADDING, max(min_y - DOMINO_ADDITIONAL_WIDTH, 0))
                bottom_right_corner = (min(max_x + DOMINO_HALF_SIZE + DOMINO_PADDING, image.shape[0] - 1), min(max_y + DOMINO_ADDITIONAL_WIDTH, image.shape[1] - 1))
            else:
                continue

            cv2.rectangle(image, top_left_corner, top_right_corner, (128, 128, 128), 3)
            cv2.rectangle(image, bottom_left_corner, bottom_right_corner, (128, 128, 128), 3)

            if is_horizontal:
                directions[row][column] = 'down'
                directions[row+1][column] = 'up'

                board[row][column] = board[row+1][column] = 0
            else:
                directions[row][column] = 'right'
                directions[row][column+1] = 'left'

                board[row][column] = board[row][column+1] = 0

            for i, circle in enumerate(circles):
                if circle is None:
                    continue

                center_x, center_y, radius = circle
                center = (center_x, center_y)
                if point_in_rect(center, top_left_corner, top_right_corner):
                    cv2.circle(image, center, radius, (0, 0, 255), -1)

                    circles[i] = None
                    board[row][column] += 1

                    board[row][column] = min(board[row][column], 6)
                elif point_in_rect(center, bottom_left_corner, bottom_right_corner):
                    cv2.circle(image, center, radius, (0, 0, 255), -1)

                    circles[i] = None
                    if is_horizontal:
                        board[row+1][column] += 1

                        board[row+1][column] = min(board[row+1][column], 6)
                    else:
                        board[row][column+1] += 1

                        board[row][column+1] = min(board[row][column+1], 6)

    return board, directions


def point_in_rect(point, left_corner, right_corner) -> bool:
    return left_corner[0] <= point[0] <= right_corner[0] and \
        left_corner[1] <= point[1] <= right_corner[1]