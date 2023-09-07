import dataclasses
import os
from enum import Enum
from typing import List

import numpy

ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')

SRC_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'src')
TESTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'tests')
CONFIG_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'config')

YOLO_MODEL_PATH = os.path.join(CONFIG_DIRECTORY, 'best.pt')

DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
TRAIN_DIRECTORY = os.path.join(DATA_DIRECTORY, 'train')

TEST_OUTPUTS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'test_outputs')

ALL_IMAGES_TASK_1 = sorted([
    "./train/Task1/22.jpg",
    "./train/Task1/25.jpg",
    "./train/Task1/15.jpg",
    "./train/Task1/12.jpg",
    "./train/Task1/13.jpg",
    "./train/Task1/18.jpg",
    "./train/Task1/19.jpg",
    "./train/Task1/01.jpg",
    "./train/Task1/24.jpg",
    "./train/Task1/06.jpg",
    "./train/Task1/14.jpg",
    "./train/Task1/20.jpg",
    "./train/Task1/04.jpg",
    "./train/Task1/17.jpg",
    "./train/Task1/08.jpg",
    "./train/Task1/21.jpg",
    "./train/Task1/10.jpg",
    "./train/Task1/16.jpg",
    "./train/Task1/02.jpg",
    "./train/Task1/05.jpg",
    "./train/Task1/03.jpg",
    "./train/Task1/11.jpg",
    "./train/Task1/09.jpg",
    "./train/Task1/07.jpg",
    "./train/Task1/23.jpg"
])
ALL_IMAGES_TASK_1 = list(map(lambda img: os.path.join(DATA_DIRECTORY, img), ALL_IMAGES_TASK_1))

ALL_IMAGES_TASK_2 = sorted([
    "./train/Task2/22.jpg",
    "./train/Task2/25.jpg",
    "./train/Task2/15.jpg",
    "./train/Task2/12.jpg",
    "./train/Task2/13.jpg",
    "./train/Task2/18.jpg",
    "./train/Task2/19.jpg",
    "./train/Task2/01.jpg",
    "./train/Task2/24.jpg",
    "./train/Task2/06.jpg",
    "./train/Task2/14.jpg",
    "./train/Task2/20.jpg",
    "./train/Task2/04.jpg",
    "./train/Task2/17.jpg",
    "./train/Task2/08.jpg",
    "./train/Task2/21.jpg",
    "./train/Task2/10.jpg",
    "./train/Task2/16.jpg",
    "./train/Task2/02.jpg",
    "./train/Task2/05.jpg",
    "./train/Task2/03.jpg",
    "./train/Task2/11.jpg",
    "./train/Task2/09.jpg",
    "./train/Task2/07.jpg",
    "./train/Task2/23.jpg"
])
ALL_IMAGES_TASK_2 = list(map(lambda img: os.path.join(DATA_DIRECTORY, img), ALL_IMAGES_TASK_2))

ALL_VIDEOS_TASK_3 = sorted([
    "./train/Task3/18.mp4",
    "./train/Task3/19.mp4",
    "./train/Task3/07.mp4",
    "./train/Task3/03.mp4",
    "./train/Task3/05.mp4",
    "./train/Task3/17.mp4",
    "./train/Task3/16.mp4",
    "./train/Task3/23.mp4",
    "./train/Task3/11.mp4",
    "./train/Task3/21.mp4",
    "./train/Task3/08.mp4",
    "./train/Task3/12.mp4",
    "./train/Task3/02.mp4",
    "./train/Task3/25.mp4",
    "./train/Task3/22.mp4",
    "./train/Task3/15.mp4",
    "./train/Task3/04.mp4",
    "./train/Task3/09.mp4",
    "./train/Task3/24.mp4",
    "./train/Task3/06.mp4",
    "./train/Task3/10.mp4",
    "./train/Task3/01.mp4",
    "./train/Task3/14.mp4",
    "./train/Task3/20.mp4",
    "./train/Task3/13.mp4"
])
ALL_VIDEOS_TASK_3 = list(map(lambda vid: os.path.join(DATA_DIRECTORY, vid), ALL_VIDEOS_TASK_3))


class BoardType(Enum):
    Simple = 0
    Classic = 1


@dataclasses.dataclass
class Settings:
    board_type: BoardType

    board_for_template_path: str
    board_template_quadrilateral: numpy.ndarray

    board_annuli: List[int]
    board_circle_center: List[int]

    board_match_max_features: int
    board_match_percent: float


def rectangle(top_left: List[int], width: int, height: int) -> List[List[int]]:
    return [
        top_left,
        [top_left[0] + width, top_left[1]],
        [top_left[0] + width, top_left[1] + height],
        [top_left[0], top_left[1] + height],
    ]


default_task1 = Settings(
    board_type=BoardType.Simple,

    board_for_template_path=os.path.join(CONFIG_DIRECTORY, 'template_task1.jpg'),
    board_template_quadrilateral=numpy.array(
        [
            [254, 1535],
            [1257, 337],
            [1994, 1405],
            [1327, 2553]
        ],
        dtype=numpy.float32
    ),

    board_annuli=[0, 33, 155, 270, 385, 500, 615, 730, 845, 960, 1200],
    board_circle_center=[1104, 1129],

    board_match_max_features=2500,
    board_match_percent=0.50
)
assert len(default_task1.board_annuli) == 11

DART_BOARD_SECTORS = 20
DART_BOARD_SECTOR_POINTS = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]

default_task2 = Settings(
    BoardType.Classic,

    board_for_template_path=os.path.join(CONFIG_DIRECTORY, 'template_task2.jpg'),
    board_template_quadrilateral=numpy.array(
        [
            [316, 1991],
            [1260, 779],
            [2015, 1737],
            [1388, 2901]
        ],
        dtype=numpy.float32
    ),

    board_annuli=[0, 40, 78, 485, 539, 810, 900],
    board_circle_center=[1084, 1089],

    board_match_max_features=2000,
    board_match_percent=0.40
)
assert len(default_task2.board_annuli) == 7


@dataclasses.dataclass
class VideoSettings:
    pass


default_task3 = VideoSettings()
