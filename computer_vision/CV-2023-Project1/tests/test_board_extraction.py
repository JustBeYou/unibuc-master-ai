import time
import unittest

import constants
import output
from vision import templates, transforms, template_matcher, patches
from dominoes import board
import settings
import logging


class BoardExtractionTestCase(unittest.TestCase):
    def test_board_extraction(self):

        bullet_template = transforms.read(settings.default.bullet_template_path)
        bullet_matcher = template_matcher.TemplateMatcher(transforms.grayscale(bullet_template), 500)

        board_for_template = transforms.read(settings.default.board_for_template_path)
        template_image = transforms.grayscale(
            templates.create(
                board_for_template,
                settings.default.board_template_quadrilateral,
                templates.TemplateType.SQUARE
            )
        )
        output.debug_output_image("Template for board extraction", template_image)

        matcher = template_matcher.TemplateMatcher(
            template_image,
            settings.default.board_match_max_features
        )

        start_time = time.time()
        for i, image_path in enumerate(constants.ALL_TRAIN_IMAGE_PATHS):
            image = transforms.read(image_path)
            match_color = matcher.match(image, settings.default.board_match_percent)
            match = transforms.grayscale(match_color)
            self.assertEqual(match.shape, template_image.shape)

            lines = patches.grid_patches(
                match,
                board.BOARD_SIZE,
                board.BOARD_SIZE,
                settings.default.board_line_thickness,
                settings.default.board_margin
            )
            match_color = transforms.draw_patches(match_color, lines)

            domino = bullet_matcher.match_all_orb(match_color, settings.default.board_match_percent)
            # match_color = transforms.draw_bullet_matches(match_color, bullets)

            # circles = transforms.get_circles(match)
            # if circles is not None:
            #     match_color = transforms.draw_circles(match_color, circles)

            name = image_path.split('/')[-1].replace('.jpg', '')
            output.debug_output_image(f"Processed board (domino match) ({i} {name})", domino)
            # output.debug_output_image(f"Processed board ({i} {name})", match_color)
            logging.debug(
                f"Time elapsed {int(time.time() - start_time)}s ({i + 1}/{len(constants.ALL_TRAIN_IMAGE_PATHS)})")

if __name__ == '__main__':
    unittest.main()
