import logging
import os
import pprint
from typing import List

import numpy

import settings
from vision import transforms, templates, template_matcher
from . import annotation, board


class Game:
    def __init__(self, name: str, source_directory: str):
        self.name: str = name
        self.images: List[numpy.ndarray] = []
        self.turns: List[str] = []
        self.annotations: List[annotation.Annotation] = []
        self.has_annotations: bool = False
        self.__load(source_directory)

    def __load(self, source_directory: str):
        moves_file = os.path.join(source_directory, f"{self.name}_moves.txt")
        if not os.path.isfile(moves_file):
            raise RuntimeError(f"Expected moves file {moves_file}, but not found.")

        with open(moves_file) as moves_file_object:
            content = moves_file_object.read().split('\n')
            content = [line.strip() for line in content]
            content = [line for line in content if line != '']
            file_names = [line.split(' ')[0] for line in content]
            self.turns = [line.split(' ')[1] for line in content]

        for file_name in file_names:
            round_image = os.path.join(source_directory, file_name)
            round_annotations = round_image.replace('.jpg', '.txt')

            self.images.append(transforms.grayscale(transforms.read(round_image)))

            if os.path.exists(round_annotations) and os.path.isfile(round_annotations):
                self.has_annotations = True
                with open(round_annotations) as round_annotations_file:
                    content = round_annotations_file.read()
                self.annotations.append(annotation.Annotation.from_string(content))

        if self.has_annotations and len(self.images) != len(self.annotations):
            raise RuntimeError("Number of images is not the same as the number of annotation files.")

    def extract_all_boards(self):
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

        all_boards = [board.Board.from_image(image, matcher) for image in self.images]
        all_boards.insert(0, board.Board())
        return all_boards

    def annotate_rounds(self, boards: List[board.Board]):
        player_score = {}

        for player in numpy.unique(self.turns):
            player_score[player] = 0

        annotations = []
        for i in range(len(boards) - 1):
            current_player = self.turns[i]
            current_board = boards[i]
            next_board = boards[i+1]
            piece = current_board.diff_one_piece(next_board)

            new_score = board.DIAMONDS[piece[0][0]][piece[0][1]] + board.DIAMONDS[piece[1][0]][piece[1][1]]
            double_domino = piece[0][2] == piece[1][2]
            if double_domino:
                new_score *= 2

            for some_player, some_score in player_score.items():
                if board.TRACK[some_score] == piece[0][2] or board.TRACK[some_score] == piece[1][2]:
                    print(f"In round {i+1}, {some_player} received a bonus of 3 points.")

                    if some_player == current_player:
                        new_score += 3
                    else:
                        player_score[some_player] += 3

            player_score[current_player] += new_score

            print(f"Round {i+1} of {current_player} with move ({piece}) and added score {new_score}")

            if len(piece) != 2:
                logging.warning(f"Round {i+1} got too many or too little diffs {piece}.")
                break

            annotations.append(annotation.Annotation.from_raw_parts(piece, new_score))

        return annotations

    def check_annotations(self, other_annotations):
        good = 0
        first_error = None
        for my_annotation, other_annotation in zip(self.annotations, other_annotations):
            # print("Left")
            # pprint.pprint(my_annotation)
            #
            # print("Right")
            # pprint.pprint(other_annotation)

            if my_annotation.same(other_annotation):
                good += 1
            else:
                first_error = [my_annotation, other_annotation]
                break
        return good, first_error