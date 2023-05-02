import pprint
import unittest

import numpy

import constants
from dominoes import game, annotation


class GameTestCase(unittest.TestCase):

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

    def test_annotate_game(self):
        for game_number in range(1, 6):
            print(f"Train - Game {game_number}")
            my_game = game.Game(str(game_number), constants.TRAIN_REGULAR_DIRECTORY)
            boards = my_game.extract_all_boards()
            annotations = my_game.annotate_rounds(boards)
            good_annotations, first_error = my_game.check_annotations(annotations)
            pprint.pprint(good_annotations)
            pprint.pprint(first_error)
            self.assertIsNone(first_error)
            self.assertEqual(len(annotations), 20)

    def test_evaluation_set(self):
        for game_number in range(1, 2):
            print(f"Evaluation - Game {game_number}")
            my_game = game.Game(str(game_number), constants.EVALUATION_FAKE_TEST_REGULAR_DIRECTORY)
            boards = my_game.extract_all_boards()
            annotations = my_game.annotate_rounds(boards)
            good_annotations, first_error = my_game.check_annotations(annotations)
            pprint.pprint(good_annotations)
            pprint.pprint(first_error)
            self.assertIsNone(first_error)
            self.assertEqual(len(annotations), 20)


if __name__ == '__main__':
    unittest.main()
