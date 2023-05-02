import unittest

import numpy

import constants
from dominoes import game, annotation


class GameTestCase(unittest.TestCase):

    def test_load_game(self):
        self.skipTest("Enable later")
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
        # self.skipTest("Too big")
        for game_number in range(1, 6):
            print(f"Game {game_number}")
            my_game = game.Game(str(game_number), constants.TRAIN_REGULAR_DIRECTORY)
            boards = my_game.extract_all_boards()
            annotations = my_game.annotate_rounds(boards)
            good_annotations, first_error = my_game.check_annotations(annotations)
            print(good_annotations, first_error)

if __name__ == '__main__':
    unittest.main()
