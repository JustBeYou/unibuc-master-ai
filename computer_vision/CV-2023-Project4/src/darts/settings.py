from typing import List, Tuple
import dataclasses
import os

import numpy

ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')

SRC_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'src')
TESTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'tests')

DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
AUXILIARY_DIRECTORY = os.path.join(DATA_DIRECTORY, 'auxiliary_images')
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

@dataclasses.dataclass
class Settings:
    dart_template_path: str
    board_for_template_path: str
    board_template_crop: List[Tuple[int, int]]
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

def gen_task1_annuli() -> List[int]:
    dif = 540 // 2 - 311 // 2
    annuli = [66 // 2, 311 // 2]
    for i in range(7):
        annuli.append(311 // 2 + dif * i)
    annuli.append(1120)
    return annuli

def gen_task2_annuli() -> List[int]:
    return [40, 156 // 2, 970 // 2, 1078 // 2, 1620 // 2, 1724 // 2]

default_task1 = Settings(
    dart_template_path=os.path.join(AUXILIARY_DIRECTORY, 'dart_template.png'),
    board_for_template_path=os.path.join(AUXILIARY_DIRECTORY, 'template_task1.jpg'),
    # board_template_quadrilateral=numpy.array(
    #     rectangle([254, 332], 1742, 2264),
    #     dtype=numpy.float32
    # ), # rectangle
    board_template_quadrilateral=numpy.array(
        [
            [254, 1535],
            [1257, 337],
            [1994, 1405],
            [1327, 2553]
        ],
        dtype=numpy.float32
    ), # circle
    board_template_crop=[(254, 332), (254 + 1742, 332 + 2264)], # no nail
    # board_template_crop=[(254, 165), (254 + 1742, 165 + 2431)], # nail included

    board_annuli=gen_task1_annuli(),
    board_circle_center=[1104, 1129],

    board_match_max_features=2500,
    board_match_percent=0.50
)

DART_BOARD_SECTORS = 20

default_task2 = Settings(
    dart_template_path=os.path.join(AUXILIARY_DIRECTORY, 'dart_template.png'),
    board_for_template_path=os.path.join(AUXILIARY_DIRECTORY, 'template_task2.jpg'),
    # board_template_quadrilateral=numpy.array(
    #     rectangle([309, 774], 1712, 2173),
    #     dtype=numpy.float32
    # ),
    board_template_quadrilateral=numpy.array(
        [
            [316, 1991],
            [1260, 779],
            [2015, 1737],
            [1388, 2901]
        ],
        dtype=numpy.float32
    ),  # circle
    board_template_crop=[(309, 774), (309 + 1712, 774 + 2173)],

    board_annuli=gen_task2_annuli(),
    board_circle_center=[1104 - 20, 1129 - 40],

    board_match_max_features=2000,
    board_match_percent=0.40
)
