import numpy as np
import os
correct_file_keyword = 'correct_movement'
incorrect_file_keyword = 'incorrect_movement'
RAW_DATA_BASE_DIR_PATH = "../data/datasets/uiprmd/extracted_raw_data"

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

