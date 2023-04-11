import dataclasses
from typing import Dict

EMPTY = -1
ROWS = list(range(1, 15 + 1))
COLUMNS = "ABCDEFGHIJKLMNO"


@dataclasses.dataclass
class Board:
    """
    A mapping of rows x columns -> domino values.
    Rows from 1 to 15, columns from A to O and values from 0 to 6.
    """
    __board: Dict[int, Dict[str, int]]

    def __init__(self):
        self.__board = {}
        for row in ROWS:
            self.__board[row] = {}
            for column in COLUMNS:
                self.__board[row][column] = EMPTY

    def __getitem__(self, item):
        return self.__board[item]
