import os
import numpy as np
from DataProcessing import DataProcessing
import re
from tqdm import tqdm
'''
The vicon_data is organized as follows

eg :- m01_s01_e_01

m -> movement, there are 10 such movements
s -> subject, there are 10 such subjects
e -> episode, there are 10 such episodes

a total of 1000 files will be there
'''

def next_line(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    n = 0
    while n < len(lines):
        yield lines[n]
        n += 1


def is_match_regex(fname, regex):
    bool_val = re.match(regex, fname) is not None
    # print(fname, bool_val)
    return bool_val


class DataExtractor:

    def __init__(self, subject_regex, exercise_regex):
        self.folder_path = None
        self.subject_match_regex = subject_regex
        self.exercise_match_regex = exercise_regex

    def is_match_regex_subject(self, fname):
        return is_match_regex(fname, self.subject_match_regex)

    def is_match_regex_exercise(self, fname):
        return is_match_regex(fname, self.exercise_match_regex)

    def get_stats_frame_len(self):
        tlen = []
        data_list = []
        label_list = []
        for fname in tqdm(os.listdir(self.folder_path)):
            if self.is_match_regex_exercise(fname.split('_')[0]):
                # arr = np.loadtxt(os.path.join(self.folder_path, fname), delimiter=',')
                arr = np.loadtxt(next_line(os.path.join(self.folder_path, fname)))
                data_list.append(arr)
                tlen.append(data_list[-1].shape[0])
                label_list.append(int(fname.split('_')[0][1:]))
        return data_list, label_list, (np.mean(tlen).astype(np.int), np.std(tlen).astype(np.int), np.median(tlen).astype(np.int))

    def extractMovementData(self, folder_path):
        dp = DataProcessing()
        self.folder_path = folder_path
        data_list, label_list, (mean, std, median) = self.get_stats_frame_len()
        # print(data_list[0].shape)
        # exit()
        filtered_data_list = []
        filtered_label_list = []
        mean = 240
        for arr, label in tqdm(zip(data_list, label_list))  :
            # if (arr.shape[0] > mean+1.25*std) or (arr.shape[0] < mean - 1.25*std):
            #     pass
            # else:
            interp_arr = dp.interpolate(arr, interp_to_frame_len=mean)
            filtered_data_list.append(interp_arr)
            filtered_label_list.append(label)
        return np.stack(filtered_data_list), np.array(filtered_label_list)

#s[0][1-8]
#s((09)|(10))
# subject_regex=r's.*'

#e[0][1-8]
#e((09)|(10))



'''
Use regex for extracting specific exercises for specific subjects.
'''
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    d = None
    # exercise_id = r'e01'
    exercise_id = 'e' + '0' * (2 - len(str(i))) + str(i)
    subject_regex = r's.*'
    d = DataExtractor(subject_regex=subject_regex, exercise_regex=exercise_id)

    '''
    These steps do the following normalization steps
    1. Mean centering at every timestep
    2. Min-Max normalization between [-1, 1]
    '''
    eps=1e-6
    correct_movements_arr, correct_label_arr = d.extractMovementData(folder_path='../raw_data/Correct Movements')
    incorrect_movements_arr, incorrect_label_arr = d.extractMovementData(folder_path='../raw_data/Incorrect Movements')

    #
    # correct_movements_arr = np.divide(correct_movements_arr - np.expand_dims(correct_movements_arr.mean(axis=1), axis=1),
    #                                   np.expand_dims(eps + np.nanstd(correct_movements_arr, axis=1), axis=1))
    #
    # incorrect_movements_arr = np.divide(incorrect_movements_arr - np.expand_dims(incorrect_movements_arr.mean(axis=1), axis=1),
    #                                     np.expand_dims(eps + np.nanstd(incorrect_movements_arr, axis=1), axis=1))
    #
    #


    correct_movements_arr = correct_movements_arr - np.expand_dims(correct_movements_arr.mean(axis=1), axis=1)
    incorrect_movements_arr = incorrect_movements_arr - np.expand_dims(incorrect_movements_arr.mean(axis=1), axis=1)

    scaling_factor = np.maximum(np.abs(np.min(correct_movements_arr)), np.max(correct_movements_arr))
    correct_movements_arr = np.divide(correct_movements_arr, scaling_factor)
    incorrect_movements_arr = np.divide(incorrect_movements_arr, scaling_factor)


    np.save('./../extracted_raw_data/correct_movement_' + exercise_id + '_arr', correct_movements_arr)
    np.save('./../extracted_raw_data/incorrect_movement_' + exercise_id + '_arr', incorrect_movements_arr)

    np.save('./../extracted_raw_data/correct_movement_' + exercise_id + '_label', correct_label_arr)
    np.save('./../extracted_raw_data/incorrect_movement_' + exercise_id + '_label', incorrect_label_arr)

    print(correct_movements_arr.shape, incorrect_movements_arr.shape)