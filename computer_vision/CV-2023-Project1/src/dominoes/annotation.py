import dataclasses

from dominoes import board


@dataclasses.dataclass
class PiecePart:
    row: int
    column: str
    value: int

    @staticmethod
    def from_string(line: str):
        parts = line.strip().split(' ')
        return PiecePart(int(parts[0][:-1]), parts[0][-1], int(parts[1]))


@dataclasses.dataclass
class Annotation:
    first: PiecePart
    second: PiecePart
    score: int

    @staticmethod
    def from_string(text: str):
        lines = text.strip().split('\n')
        return Annotation(
            PiecePart.from_string(lines[0]),
            PiecePart.from_string(lines[1]),
            int(lines[2].strip())
        )

    @staticmethod
    def from_raw_parts(parts, score):
        parts = sorted(parts, key=lambda elem: elem[:2])
        return Annotation(
            PiecePart(parts[0][0] + 1, board.COLUMNS[parts[0][1]], parts[0][2]),
            PiecePart(parts[1][0] + 1, board.COLUMNS[parts[1][1]], parts[1][2]),
            score
        )

    def same(self, annotation):
        return self.first.row == annotation.first.row and \
            self.first.column == annotation.first.column and \
            self.first.value == annotation.first.value and \
            self.second.row == annotation.second.row and \
            self.second.column == annotation.second.column and \
            self.second.value == annotation.second.value and \
            self.score == annotation.score