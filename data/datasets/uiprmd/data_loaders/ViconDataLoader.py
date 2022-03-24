import csv

import numpy as np
import os


'''
This function returns two ndarray one having correctly performed exercise data based and other incorrectly performed 
exercise data.
'''
correct_file_keyword = 'correct_movement'
incorrect_file_keyword = 'incorrect_movement'

RAW_DATA_BASE_DIR_PATH = "../data/datasets/uiprmd/extracted_raw_data"
SCORES_DATA_BASE_DIR_PATH = "../data/datasets/uiprmd/label_data"

def load_raw_data(exercise_id, RAW_DATA_BASE_DIR_PATH):
    correct_file_path = os.path.join(RAW_DATA_BASE_DIR_PATH, "correct_movement_"+exercise_id+"_arr.npy")
    incorrect_file_path = os.path.join(RAW_DATA_BASE_DIR_PATH, "incorrect_movement_" + exercise_id + "_arr.npy")
    return np.load(correct_file_path), np.load(incorrect_file_path)


def load_raw_data_with_scores(exercise_id):
    print(os.getcwd())
    correct_input = np.load(os.path.join(RAW_DATA_BASE_DIR_PATH, 'correct_movement_'+exercise_id+'_arr.npy'))
    incorrect_input = np.load(os.path.join(RAW_DATA_BASE_DIR_PATH, 'incorrect_movement_' + exercise_id + '_arr.npy'))
    f = open(os.path.join(SCORES_DATA_BASE_DIR_PATH, 'Labels_Correct_'+exercise_id+'.csv'))
    csv_f = csv.reader(f)
    Correct_Y = list(csv_f)
    # Convert the input labels into numpy arrays
    correct_label = np.asarray(Correct_Y, dtype=float)
    f = open(os.path.join(SCORES_DATA_BASE_DIR_PATH, 'Labels_Incorrect_'+exercise_id+'.csv'))
    csv_f = csv.reader(f)
    Incorrect_Y = list(csv_f)
    # Convert the input labels into numpy arrays
    incorrect_label = np.asarray(Incorrect_Y, dtype=float)
    return correct_input, correct_label, incorrect_input, incorrect_label

def load_all_raw_data_with_class_labels():
    input = []
    label = []
    for exercise_id in range(1, 11, 1):
        file_id = 'e' + (2 - len(str(exercise_id))) * '0' + str(exercise_id)
        correct_arr_path = os.path.join(RAW_DATA_BASE_DIR_PATH, correct_file_keyword + '_' + file_id + '_' + 'arr.npy')
        incorrect_arr_path = os.path.join(RAW_DATA_BASE_DIR_PATH,
                                          incorrect_file_keyword + '_' + file_id + '_' + 'arr.npy')
        correct_label_path = os.path.join(RAW_DATA_BASE_DIR_PATH,
                                          correct_file_keyword + '_' + file_id + '_' + 'label.npy')
        incorrect_label_path = os.path.join(RAW_DATA_BASE_DIR_PATH,
                                            incorrect_file_keyword + '_' + file_id + '_' + 'label.npy')

        input.append(np.load(correct_arr_path))
        input.append(np.load(incorrect_arr_path))

        label.append(np.load(correct_label_path))
        label.append(np.load(incorrect_label_path))

    combined_exercise_arr = np.vstack(input)
    combined_exercise_label = np.concatenate(label, axis=0)
    return combined_exercise_arr, combined_exercise_label

