import numpy as np
import os


RAW_DATA_BASE_DIR_PATH = "../data/datasets/kimore/processed_data"
SCORES_DATA_BASE_DIR_PATH = "../data/datasets/uiprmd/label_data"

def load_raw_data_with_labels(exercise_id, RAW_DATA_BASE_DIR_PATH="../data/datasets/kimore/processed_data"):
    correct_file_path = os.path.join(RAW_DATA_BASE_DIR_PATH, "extracted_orient_data_"+exercise_id+".npy")
    correct_label_path = os.path.join(RAW_DATA_BASE_DIR_PATH, "extracted_labels_data_" + exercise_id + ".npy")
    label = np.squeeze(np.load(correct_label_path))
    idx = np.logical_not(np.isnan(label))
    arr = np.squeeze(np.load(correct_file_path))
    return arr[idx], label[idx]


