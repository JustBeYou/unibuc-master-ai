import typing

import numpy

from src import settings
from dominoes import board
from vision import transforms, templates, template_matcher


class BonusSolver:
    def solve_images(self, labels: typing.List[str], images: typing.List[numpy.ndarray]):
        board_for_template = transforms.read(settings.default.board_for_template_path)
        template_image = transforms.grayscale(
            templates.create(
                board_for_template,
                settings.default.board_template_quadrilateral,
                templates.TemplateType.SQUARE
            )
        )
        matcher = template_matcher.TemplateMatcher(
            template_image,
            settings.default.board_match_max_features
        )

        all_boards = [board.Board.from_image(image, matcher).to_array() for image in images]

        results = {}
        for label, (some_board, some_directions) in zip(labels, all_boards):
            print("Solve", label)
            results[label] = self.solve_board(some_board, some_directions)

        return results

    def solve_board(self, some_board, some_directions):
        invalid_squares = []
        invalid_neighbours = []

        # 2x2 sliding window and valid neighbours
        for i in range(0, len(some_board) - 1):
            for j in range(0, len(some_board[0]) - 1):
                if some_board[i][j] != -1 and \
                        some_board[i][j + 1] != -1 and \
                        some_board[i + 1][j] != -1 and \
                        some_board[i + 1][j + 1] != -1:

                    print(f"Invalid square with corner {i, j}")
                    invalid_squares.append((i, j))

                elif some_board[i][j] != -1 and some_board[i+1][j] != -1 and \
                        some_board[i][j] != some_board[i+1][j] and \
                        some_directions[i][j] != 'down':

                    print(f"Invalid neighbours {(i, j), (i+1, j)}")
                    invalid_neighbours.append(((i, j), (i+1, j)))

                elif some_board[i][j] != -1 and some_board[i][j+1] != -1 and \
                        some_board[i][j] != some_board[i][j+1] and \
                        some_directions[i][j] != 'right':

                    print(f"Invalid neighbours {(i, j), (i, j + 1)}")
                    invalid_neighbours.append(((i, j), (i, j+1)))

        invalids_counts = len(invalid_squares) + len(invalid_neighbours)
        all_invalids = []
        for i, j in invalid_squares:
            all_invalids.append(board.Board.name(i, j))
            all_invalids.append(board.Board.name(i+1, j))
            all_invalids.append(board.Board.name(i, j+1))
            all_invalids.append(board.Board.name(i+1, j+1))

        for domino in invalid_neighbours:
            for i, j in domino:
                all_invalids.append(board.Board.name(i, j))

        all_invalids = sorted(all_invalids, key=lambda elem: (int(elem[:-1]), elem[-1]))

        return [str(invalids_counts)] + all_invalids