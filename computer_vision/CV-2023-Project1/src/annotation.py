import dataclasses
from typing import List


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
