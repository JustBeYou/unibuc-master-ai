import argparse
import os

import cv2

import dominoes.settings
from dominoes import game, bonus


# python src/main.py --template ./src/best_board.jpg --input ./data/fake_test_inputs/ --output ./data/fake_test_results/
# python src/evaluate_submission.py --results ./data/fake_test_results/ --truth ./data/fake_test_truth/

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--template', required=True, help='Path to a image of the board to extract a template.')
    args.add_argument('-i', '--input', required=True, help='Input directory with regular_tasks/ and bonus_task/ containing images.')
    args.add_argument('-o', '--output', required=True, help='Output directory for predictions.')

    args = args.parse_args()

    dominoes.settings.default.board_for_template_path = args.template

    regular_tasks_dir = os.path.join(args.input, 'regular_tasks')
    regular_tasks_files = os.listdir(regular_tasks_dir)

    game_names = [name.replace('_moves.txt', '') for name in regular_tasks_files if '_moves' in name]
    game_names = sorted(game_names)

    os.system(f"mkdir -p {os.path.join(args.output, 'regular_tasks')}")
    os.system(f"mkdir -p {os.path.join(args.output, 'bonus_task')}")

    print("Solving regular tasks")
    for game_name in game_names:
        print(f"Solving game {game_name}")
        try:
            my_game = game.Game(game_name, regular_tasks_dir)
            boards = my_game.extract_all_boards()
            annotations = my_game.annotate_rounds(boards)

            for prediction_name, annotation in zip(my_game.prediction_names, annotations):
                print(f"Writing output prediction to {prediction_name}")
                output_path = os.path.join(args.output, 'regular_tasks', prediction_name)
                with open(output_path, 'w') as f:
                    f.write(annotation.to_string())
                    f.flush()
        except Exception as e:
            print(f"Error while solving game {game_name}", e)

    bonus_task_dir = os.path.join(args.input, 'bonus_task')
    bonus_task_files = os.listdir(bonus_task_dir)

    files_in_dir = bonus_task_files
    input_images = [img for img in files_in_dir if '.jpg' in img]
    input_images = sorted(input_images)

    input_labels = input_images.copy()
    input_images = [os.path.join(bonus_task_dir, img) for img in input_images]
    input_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in input_images]

    print("Solving bonus task")
    solver = bonus.BonusSolver()
    results = solver.solve_images(input_labels, input_images)

    for label in results:
        try:
            prediction_name = label.replace('.jpg', '.txt')
            print(f"Writing output prediction to {prediction_name}")
            output_path = os.path.join(args.output, 'bonus_task', prediction_name)
            with open(output_path, 'w') as f:
                f.write('\n'.join(results[label]) + '\n')
                f.flush()
        except Exception as e:
            print(f"Error while solving bonus task for {label}", e)

if __name__ == "__main__":
    main()
