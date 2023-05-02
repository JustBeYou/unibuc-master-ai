import logging
import os
from typing import List

import numpy

import settings
from vision import transforms, templates, template_matcher
from . import annotation, board


class Game:
    def __init__(self, name: str, source_directory: str):
        self.name: str = name
        self.images: List[numpy.ndarray] = []
        self.annotations: List[annotation.Annotation] = []
        self.has_annotations: bool = False
        self.__load(source_directory)

    def __load(self, source_directory: str):
        moves_file = os.path.join(source_directory, f"{self.name}_moves.txt")
        if not os.path.isfile(moves_file):
            raise RuntimeError(f"Expected moves file {moves_file}, but not found.")

        round_index = 1
        while True:
            round_index_str = str(round_index).rjust(2, '0')
            file_prefix = os.path.join(source_directory, f"{self.name}_{round_index_str}")

            round_image = f"{file_prefix}.jpg"
            round_annotations = f"{file_prefix}.txt"

            if not os.path.exists(round_image) or not os.path.isfile(round_image):
                break

            self.images.append(transforms.grayscale(transforms.read(round_image)))

            if os.path.exists(round_annotations) and os.path.isfile(round_annotations):
                self.has_annotations = True
                with open(round_annotations) as round_annotations_file:
                    content = round_annotations_file.read()
                self.annotations.append(annotation.Annotation.from_string(content))

            round_index += 1

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
        annotations = []
        for i in range(len(boards) - 1):
            current_board = boards[i]
            next_board = boards[i+1]
            piece = current_board.diff_one_piece(next_board)

            if len(piece) != 2:
                logging.warning(f"Round {i+1} got too many or too little diffs {piece}.")
                break

            annotations.append(annotation.Annotation.from_raw_parts(piece))

        return annotations

    def check_annotations(self, other_annotations):
        good = 0
        first_error = None
        for my_annotation, other_annotation in zip(self.annotations, other_annotations):
            if my_annotation.same_piece(other_annotation):
                good += 1
            else:
                first_error = [my_annotation, other_annotation]
                break
        return good, first_error