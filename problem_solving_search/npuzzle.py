from typing import Tuple, List
import numpy


class Direction:
    UP = (-1, 0)
    DOWN = (+1, 0)
    LEFT = (0, -1)
    RIGHT = (0, +1)


Directions = [
    Direction.UP,
    Direction.DOWN,
    Direction.LEFT,
    Direction.RIGHT
]

DirectionStr = {
    Direction.UP: "UP",
    Direction.DOWN: "DOWN",
    Direction.LEFT: "LEFT",
    Direction.RIGHT: "RIGHT"
}

ReverseDirection = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT
}


class NPuzzle:
    n: int
    board: List[List[int]]
    empty_position: Tuple[int, int]
    total_correct: int

    def __init__(self, n, init=True) -> None:
        assert n > 1
        self.n = n
        if init:
            self._reset_board()
            self._shuffle()

    def move(self, direction: Tuple[int, int]) -> bool:
        x, y = self.empty_position

        assert self.board[x][y] == self.n**2
        assert self.can_move(direction)

        delta_x, delta_y = direction
        self._swap((x, y), (x + delta_x, y + delta_y))

    def can_move(self, direction: Tuple[int, int]) -> bool:
        x, y = self.empty_position
        delta_x, delta_y = direction
        return self._is_valid(x + delta_x, y + delta_y)

    def is_solved(self) -> bool:
        return self.total_correct == self.n**2

    def clone(self):
        clone = NPuzzle(self.n, init=False)
        clone.board = [[self.board[y][x]
                        for x in range(self.n)] for y in range(self.n)]
        clone.empty_position = self.empty_position
        clone.total_correct = self.total_correct
        return clone

    def hash(self) -> str:
        return str(self.board)

    def _reset_board(self) -> None:
        self.board = [
            [i*self.n+j+1 for j in range(self.n)] for i in range(self.n)]
        self.empty_position = (self.n-1, self.n-1)
        self.total_correct = self.n**2

    def _shuffle(self) -> None:
        """
        Fisher–Yates_shuffle
        """

        while True:
            for k in range(self.n**2-1, 0, -1):
                q = numpy.random.randint(0, k)

                x_k, y_k = k // self.n, k % self.n
                x_q, y_q = q // self.n, q % self.n

                self._swap((x_k, y_k), (x_q, y_q))

            if self._is_solvable():
                break

    def _is_solvable(self):
        """
        https://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/
        If N is odd, then puzzle instance is solvable if number of inversions is even in the input state.
        If N is even, puzzle instance is solvable if
            the blank is on an even row counting from the bottom (second-last, fourth-last, etc.) and number of inversions is odd.
            the blank is on an odd row counting from the bottom (last, third-last, fifth-last, etc.) and number of inversions is even.
        For all other cases, the puzzle instance is not solvable.
        """

        inversions = self._inversions()

        if self.n % 2 == 1 and inversions % 2 == 0:
            return True

        x, _ = self.empty_position

        if (self.n - x) % 2 == 0 and inversions % 2 == 1:
            return True

        if (self.n - x) % 2 == 1 and inversions % 2 == 0:
            return True

        return False

    def _inversions(self):
        """
        https://mathworld.wolfram.com/InversionVector.html
        https://mathworld.wolfram.com/PermutationInversion.html
        """

        inversions = 0
        l = [y for x in self.board for y in x]
        for i in range(1, self.n**2):
            if l[i] == self.n**2:
                continue

            for j in range(0, i):
                if l[j] == self.n**2:
                    continue

                if l[j] > l[i]:
                    inversions += 1

        return inversions

    def _swap(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        x_a, y_a = a
        x_b, y_b = b

        self._update_correct_for_swap(a, b)
        self.board[x_a][y_a], self.board[x_b][y_b] = self.board[x_b][y_b], self.board[x_a][y_a]

        if self.board[x_a][y_a] == self.n**2:
            self.empty_position = a
        elif self.board[x_b][y_b] == self.n**2:
            self.empty_position = b

    def _update_correct_for_swap(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        x_a, y_a = a
        x_b, y_b = b

        v_a = x_a*self.n+y_a+1
        v_b = x_b*self.n+y_b+1

        if self.board[x_a][y_a] == v_a:
            self.total_correct -= 1
        elif self.board[x_a][y_a] == v_b:
            self.total_correct += 1

        if self.board[x_b][y_b] == v_b:
            self.total_correct -= 1
        elif self.board[x_b][y_b] == v_a:
            self.total_correct += 1

    def _is_valid(self, x: int, y: int) -> None:
        return x >= 0 and y >= 0 and x < self.n and y < self.n

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        row_format = ("{:>" + str(len(str(self.n**2)) + 1) + "}") * self.n
        rows = []
        for row in self.board:
            rows.append(row_format.format(*map(str, row)))
        return '\n'.join(rows)
