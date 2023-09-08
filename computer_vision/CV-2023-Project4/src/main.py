import argparse
import logging
import os

from darts import settings
from darts.detector import Detector
from darts.video_detector import detect_in_videos


# python src/main.py --input ./data/evaluation/fake_test --output ./data/evaluation/fake_results

def main():
    logging.basicConfig(level=logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', required=True,
                      help='Input directory with regular_tasks/ and bonus_task/ containing images.')
    args.add_argument('-o', '--output', required=True, help='Output directory for predictions.')
    args.add_argument('--debug', required=False, default=False)
    args = args.parse_args()

    prepare_output_dir(args)

    logging.info("Loading the detection model will take some time, so processing the first image will be slow.")
    logging.info("Please be patient.")

    detector_task1 = Detector(settings.default_task1, debug_mode=args.debug)
    logging.info("Initialized detector for Task 1. Model weights will be loaded lazily.")
    task1_files = list_files_in_dir(os.path.join(args.input, "Task1"))
    logging.info(f"Loaded {len(task1_files)} images for Task 1.")
    detections = detector_task1.detect_arrowheads(task1_files)
    output_answers(args, "Task1", task1_files, detections)

    detector_task2 = Detector(settings.default_task2, debug_mode=args.debug)
    logging.info("Initialized detector for Task 2.")
    task2_files = list_files_in_dir(os.path.join(args.input, "Task2"))
    logging.info(f"Loaded {len(task2_files)} images for Task 2.")
    detections = detector_task2.detect_arrowheads(task2_files)
    output_answers(args, "Task2", task2_files, detections)

    task3_files = list_files_in_dir(os.path.join(args.input, "Task3"))
    logging.info(f"Loaded {len(task3_files)} videos for Task 3.")
    detections = detect_in_videos(task3_files, debug_mode=args.debug)
    output_answers_3(args, "Task3", task3_files, detections)

def prepare_output_dir(args):
    os.system(f'mkdir -p {args.output}/Task1')
    os.system(f'mkdir -p {args.output}/Task2')
    os.system(f'mkdir -p {args.output}/Task3')


def list_files_in_dir(folder):
    return sorted(list(map(lambda entry: os.path.join(folder, entry), filter(lambda entry: '.mp4' in entry or '.jpg' in entry or '.jpeg' in entry,os.listdir(folder)))))


def output_answers(args, task, files, detections):
    for file, detection in zip(files, detections):
        output_name = os.path.join(args.output, task, replace_img_with_txt(file.split('/')[-1]))
        with open(output_name, 'w') as fout:
            fout.write(str(len(detection)) + "\n")
            fout.write('\n'.join(detection) + "\n")
            fout.flush()

def output_answers_3(args, task, files, detections):
    for file, detection in zip(files, detections):
        output_name = os.path.join(args.output, task, replace_img_with_txt(file.split('/')[-1]))
        with open(output_name, 'w') as fout:
            fout.write(detection + "\n")
            fout.flush()


def replace_img_with_txt(name):
    ending = '_predicted.txt'
    return name.replace('.jpg', ending).replace('.jpeg', ending).replace('.png', ending).replace('.mp4', ending)


if __name__ == "__main__":
    main()
