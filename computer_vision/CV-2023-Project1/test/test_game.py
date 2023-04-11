import unittest

import numpy

import annotation
import constants
import game


class MyTestCase(unittest.TestCase):

    def test_load_game(self):
        my_game = game.Game('1', constants.TRAIN_REGULAR_DIRECTORY)

        self.assertEqual(len(my_game.images), 20)
        self.assertEqual(type(my_game.images[0]), numpy.ndarray)

        self.assertEqual(my_game.has_annotations, True)
        self.assertEqual(len(my_game.annotations), 20)

        self.assertEqual(my_game.annotations[0], annotation.Annotation(
            annotation.PiecePart(8, 'H', 6),
            annotation.PiecePart(8, 'I', 1),
            0
        ))
        self.assertEqual(my_game.annotations[14], annotation.Annotation(
            annotation.PiecePart(13, 'N', 3),
            annotation.PiecePart(13, 'O', 5),
            6
        ))


if __name__ == '__main__':
    unittest.main()
