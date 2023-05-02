import os.path

ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

SRC_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'src')
TESTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'tests')

DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
TRAIN_REGULAR_DIRECTORY = os.path.join(DATA_DIRECTORY, 'train', 'regular_tasks')
TEST_OUTPUTS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'test_outputs')

BOARDS_AND_DOMINOES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'board+dominoes')

WRITE_OUTPUTS = True

ALL_TRAIN_IMAGE_PATHS = sorted([
    "./board+dominoes/15.jpg",
    "./board+dominoes/12.jpg",
    "./board+dominoes/13.jpg",
    "./board+dominoes/01.jpg",
    "./board+dominoes/06.jpg",
    "./board+dominoes/14.jpg",
    "./board+dominoes/04.jpg",
    "./board+dominoes/08.jpg",
    "./board+dominoes/10.jpg",
    "./board+dominoes/02.jpg",
    "./board+dominoes/05.jpg",
    "./board+dominoes/03.jpg",
    "./board+dominoes/11.jpg",
    "./board+dominoes/09.jpg",
    "./board+dominoes/07.jpg",
    "./evaluation/fake_test/regular_task/1_05.jpg",
    "./evaluation/fake_test/regular_task/1_12.jpg",
    "./evaluation/fake_test/regular_task/1_06.jpg",
    "./evaluation/fake_test/regular_task/1_09.jpg",
    "./evaluation/fake_test/regular_task/1_19.jpg",
    "./evaluation/fake_test/regular_task/1_15.jpg",
    "./evaluation/fake_test/regular_task/1_16.jpg",
    "./evaluation/fake_test/regular_task/1_20.jpg",
    "./evaluation/fake_test/regular_task/1_04.jpg",
    "./evaluation/fake_test/regular_task/1_08.jpg",
    "./evaluation/fake_test/regular_task/1_03.jpg",
    "./evaluation/fake_test/regular_task/1_01.jpg",
    "./evaluation/fake_test/regular_task/1_07.jpg",
    "./evaluation/fake_test/regular_task/1_14.jpg",
    "./evaluation/fake_test/regular_task/1_13.jpg",
    "./evaluation/fake_test/regular_task/1_10.jpg",
    "./evaluation/fake_test/regular_task/1_02.jpg",
    "./evaluation/fake_test/regular_task/1_17.jpg",
    "./evaluation/fake_test/regular_task/1_18.jpg",
    "./evaluation/fake_test/regular_task/1_11.jpg",
    "./evaluation/fake_test/bonus_task/01.jpg",
    "./evaluation/fake_test/bonus_task/04.jpg",
    "./evaluation/fake_test/bonus_task/02.jpg",
    "./evaluation/fake_test/bonus_task/05.jpg",
    "./evaluation/fake_test/bonus_task/03.jpg",
    "./train/regular_tasks/2_14.jpg",
    "./train/regular_tasks/5_17.jpg",
    "./train/regular_tasks/4_18.jpg",
    "./train/regular_tasks/2_01.jpg",
    "./train/regular_tasks/2_20.jpg",
    "./train/regular_tasks/5_04.jpg",
    "./train/regular_tasks/2_03.jpg",
    "./train/regular_tasks/2_18.jpg",
    "./train/regular_tasks/3_03.jpg",
    "./train/regular_tasks/4_01.jpg",
    "./train/regular_tasks/4_13.jpg",
    "./train/regular_tasks/3_08.jpg",
    "./train/regular_tasks/3_17.jpg",
    "./train/regular_tasks/3_04.jpg",
    "./train/regular_tasks/5_16.jpg",
    "./train/regular_tasks/4_08.jpg",
    "./train/regular_tasks/1_05.jpg",
    "./train/regular_tasks/2_19.jpg",
    "./train/regular_tasks/1_12.jpg",
    "./train/regular_tasks/4_10.jpg",
    "./train/regular_tasks/2_06.jpg",
    "./train/regular_tasks/4_20.jpg",
    "./train/regular_tasks/2_13.jpg",
    "./train/regular_tasks/2_11.jpg",
    "./train/regular_tasks/4_11.jpg",
    "./train/regular_tasks/3_11.jpg",
    "./train/regular_tasks/3_12.jpg",
    "./train/regular_tasks/2_16.jpg",
    "./train/regular_tasks/1_06.jpg",
    "./train/regular_tasks/1_09.jpg",
    "./train/regular_tasks/4_16.jpg",
    "./train/regular_tasks/1_19.jpg",
    "./train/regular_tasks/1_15.jpg",
    "./train/regular_tasks/2_10.jpg",
    "./train/regular_tasks/2_15.jpg",
    "./train/regular_tasks/5_08.jpg",
    "./train/regular_tasks/5_12.jpg",
    "./train/regular_tasks/2_09.jpg",
    "./train/regular_tasks/5_15.jpg",
    "./train/regular_tasks/3_16.jpg",
    "./train/regular_tasks/1_16.jpg",
    "./train/regular_tasks/3_10.jpg",
    "./train/regular_tasks/3_20.jpg",
    "./train/regular_tasks/1_20.jpg",
    "./train/regular_tasks/1_04.jpg",
    "./train/regular_tasks/5_09.jpg",
    "./train/regular_tasks/3_15.jpg",
    "./train/regular_tasks/5_19.jpg",
    "./train/regular_tasks/1_08.jpg",
    "./train/regular_tasks/1_03.jpg",
    "./train/regular_tasks/4_14.jpg",
    "./train/regular_tasks/5_06.jpg",
    "./train/regular_tasks/3_19.jpg",
    "./train/regular_tasks/3_02.jpg",
    "./train/regular_tasks/3_14.jpg",
    "./train/regular_tasks/2_12.jpg",
    "./train/regular_tasks/4_12.jpg",
    "./train/regular_tasks/4_15.jpg",
    "./train/regular_tasks/1_01.jpg",
    "./train/regular_tasks/3_01.jpg",
    "./train/regular_tasks/4_04.jpg",
    "./train/regular_tasks/3_18.jpg",
    "./train/regular_tasks/5_13.jpg",
    "./train/regular_tasks/5_03.jpg",
    "./train/regular_tasks/5_05.jpg",
    "./train/regular_tasks/4_02.jpg",
    "./train/regular_tasks/5_20.jpg",
    "./train/regular_tasks/1_07.jpg",
    "./train/regular_tasks/1_14.jpg",
    "./train/regular_tasks/4_19.jpg",
    "./train/regular_tasks/4_06.jpg",
    "./train/regular_tasks/1_13.jpg",
    "./train/regular_tasks/2_07.jpg",
    "./train/regular_tasks/5_14.jpg",
    "./train/regular_tasks/5_11.jpg",
    "./train/regular_tasks/1_10.jpg",
    "./train/regular_tasks/5_02.jpg",
    "./train/regular_tasks/5_18.jpg",
    "./train/regular_tasks/2_02.jpg",
    "./train/regular_tasks/2_17.jpg",
    "./train/regular_tasks/2_05.jpg",
    "./train/regular_tasks/3_13.jpg",
    "./train/regular_tasks/4_17.jpg",
    "./train/regular_tasks/3_05.jpg",
    "./train/regular_tasks/2_04.jpg",
    "./train/regular_tasks/1_02.jpg",
    "./train/regular_tasks/5_07.jpg",
    "./train/regular_tasks/3_06.jpg",
    "./train/regular_tasks/1_17.jpg",
    "./train/regular_tasks/3_07.jpg",
    "./train/regular_tasks/1_18.jpg",
    "./train/regular_tasks/4_05.jpg",
    "./train/regular_tasks/5_10.jpg",
    "./train/regular_tasks/4_09.jpg",
    "./train/regular_tasks/2_08.jpg",
    "./train/regular_tasks/3_09.jpg",
    "./train/regular_tasks/4_07.jpg",
    "./train/regular_tasks/4_03.jpg",
    "./train/regular_tasks/1_11.jpg",
    "./train/regular_tasks/5_01.jpg",
    "./train/bonus_task/01.jpg",
    "./train/bonus_task/06.jpg",
    "./train/bonus_task/04.jpg",
    "./train/bonus_task/08.jpg",
    "./train/bonus_task/10.jpg",
    "./train/bonus_task/02.jpg",
    "./train/bonus_task/05.jpg",
    "./train/bonus_task/03.jpg",
    "./train/bonus_task/09.jpg",
    "./train/bonus_task/07.jpg"
])
# ALL_TRAIN_IMAGE_PATHS = [
#     "./evaluation/fake_test/bonus_task/04.jpg",
# ]
ALL_TRAIN_IMAGE_PATHS = [os.path.join(DATA_DIRECTORY, image) for image in ALL_TRAIN_IMAGE_PATHS]
