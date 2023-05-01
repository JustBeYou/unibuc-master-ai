import os.path
import unittest

import numpy as np
import time
import logging

import constants
import output
from vision import transforms, patches, templates, template_matcher
from dominoes import board
import settings

class DominoExtractionTestCase(unittest.TestCase):
    def test_patches_difference_one(self):
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

        old = transforms.grayscale(transforms.read(os.path.join(constants.TRAIN_REGULAR_DIRECTORY, '1_01.jpg')))
        new = transforms.grayscale(transforms.read(os.path.join(constants.TRAIN_REGULAR_DIRECTORY, '1_02.jpg')))

        old = matcher.match(old, settings.default.board_match_percent)
        new = matcher.match(new, settings.default.board_match_percent)

        board_patches = patches.grid_patches(
                template_image,
                board.BOARD_SIZE,
                board.BOARD_SIZE,
                settings.default.board_line_thickness,
                settings.default.board_margin
            )
        changes, differences = patches.diffs(old, new, board_patches, settings.default.patch_difference_threshold)

        self.assertEqual(len(changes), 2)
        self.assertEqual(board_patches[changes[0]].x, 5)
        self.assertEqual(board_patches[changes[1]].x, 6)
        self.assertEqual(board_patches[changes[0]].y, 7)
        self.assertEqual(board_patches[changes[1]].y, 7)

        diff_image = np.ones(template_image.shape)
        for patch, diff in zip(board_patches, differences):
            left_x, left_y = patch.top_left.as_tuple()
            right_x, right_y = patch.bottom_right.as_tuple()

            diff_image[left_y:right_y, left_x:right_x] *= diff

        output.debug_output_image("Diff image", diff_image)

        only_changed = [board_patches[i] for i in changes]
        board_with_changes = transforms.draw_patches(new, only_changed)

        output.debug_output_image("New domino highlight", board_with_changes)

    def test_patches_difference_all(self):
        self.skipTest('Too big')
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

        old = transforms.grayscale(transforms.read(os.path.join(constants.BOARDS_AND_DOMINOES_DIRECTORY, '01.jpg')))
        old = matcher.match(old, settings.default.board_match_percent)

        board_patches = patches.grid_patches(
                template_image,
                board.BOARD_SIZE,
                board.BOARD_SIZE,
                settings.default.board_line_thickness,
                settings.default.board_margin
        )

        start_time = time.time()
        for i, image_path in enumerate(constants.ALL_TRAIN_IMAGE_PATHS):
            new = transforms.read(image_path)
            new = matcher.match(new, settings.default.board_match_percent)
            self.assertEqual(new.shape, template_image.shape)

            changes, differences = patches.diffs(old, new, board_patches, settings.default.patch_difference_threshold)
            only_changed = [board_patches[i] for i in changes]
            dominoes_recognized = transforms.draw_patches(new, only_changed)

            name = image_path.split('/')[-1].replace('.jpg', '')
            output.debug_output_image(f"Dominoes recognized ({i} {name})", dominoes_recognized)
            logging.debug(
                f"Time elapsed {int(time.time() - start_time)}s ({i + 1}/{len(constants.ALL_TRAIN_IMAGE_PATHS)})")
if __name__ == '__main__':
    unittest.main()
