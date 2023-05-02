import os
import pprint
import unittest

import cv2

import constants
from dominoes import bonus


class GameTestCase(unittest.TestCase):

    def test_invalid_domino(self):
        self.invalid_domino(constants.TRAIN_BONUS_DIRECTORY)
        self.invalid_domino(constants.EVALUATION_FAKE_TEST_BONUS_DIRECTORY)

    def invalid_domino(self, directory):
        files_in_dir = os.listdir(directory)
        input_images = [img for img in files_in_dir if '.jpg' in img]
        input_images = sorted(input_images)

        input_labels = input_images.copy()
        input_images = [os.path.join(directory, img) for img in input_images]
        input_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in input_images]

        solver = bonus.BonusSolver()
        results = solver.solve_images(input_labels, input_images)

        annotations = [ann for ann in files_in_dir if '.txt' in ann]
        annotations = [os.path.join(directory, ann) for ann in annotations]
        annotations = sorted(annotations)
        annotations = [[annotation.replace('.txt', '.jpg').split('/')[-1], annotation] for annotation in annotations]
        for i in range(len(annotations)):
            with open(annotations[i][1]) as f:
                annotations[i][1] = f.read().strip()

        for label, annotation in annotations:
            print(label, results[label], annotation.split('\n'))
            self.assertListEqual(results[label], [x.strip() for x in annotation.split('\n')])

if __name__ == '__main__':
    unittest.main()
