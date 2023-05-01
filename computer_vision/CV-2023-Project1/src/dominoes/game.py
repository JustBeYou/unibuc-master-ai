import os
from typing import List

import numpy

from vision import transforms
from . import annotation


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
