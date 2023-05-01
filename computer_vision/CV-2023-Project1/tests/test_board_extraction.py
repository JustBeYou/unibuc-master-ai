import time
import unittest

import constants
import output
from vision import templates, transforms, template_matcher, patches, morphology
from dominoes import board
import settings
import logging


class BoardExtractionTestCase(unittest.TestCase):
    def test_board_extraction(self):

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

            patches_list = patches.grid_patches(
                match,
                board.BOARD_SIZE,
                board.BOARD_SIZE,
                settings.default.board_line_thickness,
                settings.default.board_margin
            )
            circles = morphology.get_domino_circles_from_patches(match, patches_list)
            mid_lines = morphology.get_domino_mid_lines(match)

            filtered_for_lines = transforms.filter_for_domino_mid_lines(match_color)
            filtered_for_lines = transforms.draw_circles(filtered_for_lines, circles)

            # match_color = transforms.draw_patches(match_color, patches_list)

            filter_mid_lines = morphology.filter_mid_lines(filtered_for_lines, mid_lines)

            match_color = transforms.colored_bgr(filtered_for_lines)
            match_color = transforms.draw_contours(match_color, filter_mid_lines)
            # match_color = transforms.draw_circles(match_color, circles)

            name = image_path.split('/')[-1].replace('.jpg', '')
            output.debug_output_image(f"Processed board ({i} {name})", match_color)
            logging.debug(
                f"Time elapsed {int(time.time() - start_time)}s ({i + 1}/{len(constants.ALL_TRAIN_IMAGE_PATHS)})")


if __name__ == '__main__':
    unittest.main()
