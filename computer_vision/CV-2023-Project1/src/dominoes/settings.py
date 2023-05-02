import dataclasses
import os

import numpy

ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')

SRC_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'src')
TESTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'tests')

DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
TRAIN_REGULAR_DIRECTORY = os.path.join(DATA_DIRECTORY, 'train', 'regular_tasks')
TEST_OUTPUTS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'test_outputs')

BOARDS_AND_DOMINOES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'board+dominoes')


@dataclasses.dataclass
class Settings:
    board_for_template_path: str
    board_template_quadrilateral: numpy.ndarray

    board_match_max_features: int
    board_match_percent: float

    board_line_thickness: int
    board_margin: int


default = Settings(
    board_for_template_path=os.path.join(BOARDS_AND_DOMINOES_DIRECTORY, '01.jpg'),
    board_template_quadrilateral=numpy.array([[592, 1063], [2491, 1019], [2572, 2948], [629, 3020]],
                                             dtype=numpy.float32),

    board_match_max_features=2000,
    board_match_percent=0.15,

    board_line_thickness=5,
    board_margin=13
)
