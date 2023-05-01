import dataclasses
from typing import List

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

    def __init__(self):
        self.__board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    @staticmethod
    def from_image(self):
        board = Board()
        return board

    def __getitem__(self, item):
        return self.__board[item]
