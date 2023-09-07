import numpy as np


def evaluate_results_task1(predictions_path, ground_truth_path, verbose=0):
    total_correct_number_darts = 0
    total_correct_number_darts_values = 0

    for i in range(1, 26):
        correct_number_darts = 0
        correct_number_darts_values = 0
        nb_matches_darts = 0

        try:
            if (i < 10):
                name = '0' + str(i)
            else:
                name = str(i)

            filename_predictions = predictions_path + name + ".txt"
            filename_ground_truth = ground_truth_path + name + ".txt"

            p = open(filename_predictions, "rt")
            gt = open(filename_ground_truth, "rt")

            number_darts = 1

            # read the first line - number of darts
            p_number_darts = int(p.readline())
            gt_number_darts = int(gt.readline())

            if (p_number_darts != gt_number_darts):
                print("BAD COUNT", filename_predictions)
                number_darts = 0

            total_correct_number_darts = total_correct_number_darts + number_darts

            p_darts_values = np.zeros(int(p_number_darts))

            for j in range(0, p_number_darts):
                p_darts_values[j] = p.readline()

            gt_darts_values = np.zeros(int(gt_number_darts))
            for j in range(0, gt_number_darts):
                gt_darts_values[j] = gt.readline()

            print(p_darts_values, gt_darts_values)
            for a, b in zip(p_darts_values, gt_darts_values):
                if a != b:
                    print("BAD DART", filename_predictions)
                    break

            nb_matches_darts = 0

            if p_number_darts > 3:
                p_number_darts = 3

            visited_p = np.zeros(p_number_darts)

            for idx_gt in range(0, gt_number_darts):

                current_dart_value = gt_darts_values[idx_gt]

                for idx_p in range(0, p_number_darts):
                    if ((current_dart_value == p_darts_values[idx_p]) and (visited_p[idx_p] == 0)):
                        visited_p[idx_p] = 1
                        nb_matches_darts = nb_matches_darts + 1
                        break

            total_correct_number_darts_values = total_correct_number_darts_values + nb_matches_darts

            p.close()
            gt.close()

        except Exception as e:
            print("Error", e)

        if verbose:
            print("Task 1 - Counting number darts + their values: for test example number ", str(i),
                  " the prediction is :", (1 - number_darts) * "in" + "correct for number of darts and guesses ",
                  (nb_matches_darts), " values of darts" + "\n")

        points = total_correct_number_darts * 0.04 + total_correct_number_darts_values * 0.02

    return total_correct_number_darts, total_correct_number_darts_values, points


def evaluate_results_task2(predictions_path, ground_truth_path, verbose=0):
    total_correct_number_darts = 0
    total_correct_number_darts_values = 0
    total_correct_number_darts_flags = 0

    nb_matches_darts_value = 0
    nb_matches_darts_flag = 0

    number_darts = 0

    for i in range(1, 26):

        try:
            if (i < 10):
                name = '0' + str(i)
            else:
                name = str(i)

            filename_predictions = predictions_path + name + ".txt"
            filename_ground_truth = ground_truth_path + name + ".txt"

            p = open(filename_predictions, "rt")
            gt = open(filename_ground_truth, "rt")

            number_darts = 1

            # read the first line - number of darts
            p_number_darts = int(p.readline())
            gt_number_darts = int(gt.readline())

            if (p_number_darts != gt_number_darts):
                number_darts = 0

            total_correct_number_darts = total_correct_number_darts + number_darts

            nb_matches_darts_value = 0
            nb_matches_darts_flag = 0

            p_darts_values = np.zeros(p_number_darts)
            p_darts_flags = []

            if p_number_darts > 3:
                p_number_darts = 3

            for j in range(0, p_number_darts):
                current_dart_value_flag = p.readline()

                flag = current_dart_value_flag[0]
                p_darts_flags.append(flag)
                value = current_dart_value_flag[1:]
                p_darts_values[j] = value

            gt_darts_values = np.zeros(gt_number_darts)
            gt_darts_flags = []

            for j in range(0, gt_number_darts):
                current_dart_value_flag = gt.readline()

                flag = current_dart_value_flag[0]
                gt_darts_flags.append(flag)
                value = current_dart_value_flag[1:]
                gt_darts_values[j] = value

            visited_p = np.zeros(p_number_darts)

            for idx_gt in range(0, gt_number_darts):

                current_dart_value = gt_darts_values[idx_gt]

                for idx_p in range(0, p_number_darts):
                    if ((current_dart_value == p_darts_values[idx_p]) and (visited_p[idx_p] == 0)):
                        visited_p[idx_p] = 1
                        nb_matches_darts_value = nb_matches_darts_value + 1
                        break

            visited_p = np.zeros(p_number_darts)

            for idx_gt in range(0, gt_number_darts):

                current_dart_flag = gt_darts_flags[idx_gt]

                for idx_p in range(0, p_number_darts):
                    if ((current_dart_flag == p_darts_flags[idx_p]) and (visited_p[idx_p] == 0)):
                        visited_p[idx_p] = 1
                        nb_matches_darts_flag = nb_matches_darts_flag + 1
                        break

            total_correct_number_darts_values = total_correct_number_darts_values + nb_matches_darts_value
            total_correct_number_darts_flags = total_correct_number_darts_flags + nb_matches_darts_flag

            p.close()
            gt.close()

        except Exception as e:
            print("Error", e)

        if verbose:
            print("Task 2 - Counting number darts + their values and flags: for test example number ", str(i),
                  " the prediction is :", (1 - number_darts) * "in" + "correct for number of darts, guesses ",
                  (nb_matches_darts_value), " values of darts and ", (nb_matches_darts_flag), " flags of darts " + "\n")

        points = total_correct_number_darts * 0.03 + total_correct_number_darts_values * 0.01 + total_correct_number_darts_flags * 0.005

    return total_correct_number_darts, total_correct_number_darts_values, total_correct_number_darts_flags, points


def evaluate_results_task3(predictions_path, ground_truth_path, verbose=0):
    total_correct_number_darts_values = 0
    total_correct_number_darts_flags = 0

    for i in range(1, 26):

        correct_flag = 0
        correct_value = 0

        try:
            if (i < 10):
                name = '0' + str(i)
            else:
                name = str(i)

            filename_predictions = predictions_path + name + ".txt"
            filename_ground_truth = ground_truth_path + name + ".txt"

            p = open(filename_predictions, "rt")
            gt = open(filename_ground_truth, "rt")

            p_dart_value_flag = p.readline()
            p_flag = p_dart_value_flag[0]
            p_value = p_dart_value_flag[1:]

            gt_dart_value_flag = gt.readline()
            gt_flag = gt_dart_value_flag[0]
            gt_value = gt_dart_value_flag[1:]

            if (p_flag == gt_flag):
                correct_flag = 1
                total_correct_number_darts_values = total_correct_number_darts_values + 1

            if (p_value == gt_value):
                correct_value = 1
                total_correct_number_darts_flags = total_correct_number_darts_flags + 1

            p.close()
            gt.close()

        except Exception as e:
            print("Error", e)

        if verbose:
            print("Prediction: ", p_dart_value_flag)
            print("Ground-truth: ", gt_dart_value_flag)
            print("Task 3 - For test example number ", str(i), " the prediction is :",
                  (1 - correct_flag) * "in" + "correct for flag, and  ",
                  (1 - correct_value) * "in" + "correct for value " + "\n")

        points = total_correct_number_darts_values * 0.03 + total_correct_number_darts_flags * 0.03

    return total_correct_number_darts_values, total_correct_number_darts_flags, points


# change this on your machine
predictions_path_root = "data/evaluation/fake_results/"
ground_truth_path_root = "data/train/"

# task1
verbose = 1
predictions_path = predictions_path_root + "Task1/"
ground_truth_path = ground_truth_path_root + "Task1/"
total_correct_number_darts, total_correct_number_darts_values, points_task1 = evaluate_results_task1(predictions_path,
                                                                                                     ground_truth_path,
                                                                                                     verbose)

print("Task 1 = ", points_task1)

# task2
verbose = 1
predictions_path = predictions_path_root + "Task2/"
ground_truth_path = ground_truth_path_root + "Task2/"
total_correct_number_darts, total_correct_number_darts_values, total_correct_number_darts_flags, points_task2 = evaluate_results_task2(
    predictions_path, ground_truth_path, verbose)

print("Task 2 = ", points_task2)

# task3
verbose = 1
predictions_path = predictions_path_root + "Task3/"
ground_truth_path = ground_truth_path_root + "Task3/"
ttotal_correct_number_darts_values, total_correct_number_darts_flags, points_task3 = evaluate_results_task3(
    predictions_path, ground_truth_path, verbose)
print("Task 3 = ", points_task3)

print("\n\nTask 1 = ", points_task1, "\nTask 2 = ", points_task2, "\nTask 3 = ", points_task3,
      "\nTo to add 0.5 points ex officio")
