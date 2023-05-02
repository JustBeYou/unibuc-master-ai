import dataclasses
from typing import List

import numpy

from src import settings

from vision import template_matcher, transforms, patches, morphology, extract

EMPTY = -1
BOARD_SIZE = 15
N = BOARD_SIZE - 1
COLUMNS = "ABCDEFGHIJKLMNO"

DIAMONDS = {}


@dataclasses.dataclass
class Board:
    """
    A mapping of rows x columns (15x15) -> domino values.
    Rows from 1 to 15, columns from A to O and values from 0 to 6.
    """
    __board: List[List[int]]

    def __init__(self, custom_board=None):
        if custom_board is None:
            self.__board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        else:
            self.__board = custom_board

    @staticmethod
    def from_image(image: numpy.ndarray, matcher: template_matcher.TemplateMatcher):
        match_color = matcher.match(image, settings.default.board_match_percent)
        match = transforms.grayscale(match_color)

        patches_list, patch_x_size, patch_y_size = patches.grid_patches(
            match,
            BOARD_SIZE,
            BOARD_SIZE,
            settings.default.board_line_thickness,
            settings.default.board_margin
        )
        circles = morphology.get_domino_circles_from_patches(match, patches_list)
        mid_lines = morphology.get_domino_mid_lines(match)

        # Make the image black and white
        image_with_filters = transforms.filter_for_domino_mid_lines(match_color)
        # Draw the black circles of the dominos on the image
        image_with_filters = transforms.draw_circles(image_with_filters, circles)
        # Filter the mid lines of the dominoes based on their neighboring regions
        mid_lines_filtered = morphology.filter_mid_lines(image_with_filters, mid_lines)

        lines_in_grid, horizontal_grid, vertical_grid = morphology.lines_to_grid(
            mid_lines_filtered,
            patches.patches_list_to_matrix(patches_list, BOARD_SIZE, BOARD_SIZE),
            patch_x_size,
            patch_y_size,
            settings.default.board_margin,
            BOARD_SIZE,
            BOARD_SIZE,
        )

        the_board = extract.create_board_from(
            horizontal_grid, vertical_grid, circles, BOARD_SIZE,
            BOARD_SIZE, match_color
        )

        return Board(the_board)

    def __getitem__(self, item):
        return self.__board[item]

    @staticmethod
    def name(i, j):
        return f"{i+1}{COLUMNS[j]}"

    def diff_one_piece(self, other_board):
        diff_piece = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.__board[i][j] != other_board[i][j]:
                    diff_piece.append((i, j, other_board[i][j]))

        return diff_piece
