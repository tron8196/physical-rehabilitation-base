import os.path

import numpy as np


class DataStats:

    def __init__(self, root_path='./raw_data'):
        self.Abbrevations = {'E':'Expert','NE':'NotExpert','BP':'BackPain','P':'Parkinson','S':'Stroke'}
        self.root_path = root_path

    def print_stats(self, source_mean_len_dict):
        for key, val in source_mean_len_dict.items():
            print('Stats for '+ key)
            for i, ex_stat_list in enumerate(val):
                print('Stats for Exercise '+str(i+1))
                print(ex_stat_list)
            print("#################################################")

    def get_stats_frame_len(self, source_select_list=['E', 'S']):
        source_mean_len_dict = {}
        for source in source_select_list:
            source_name = self.Abbrevations[source]
            path_to_look = os.path.join(self.root_path, source_name, 'Position_Data')
            tlen = [[], [], [], [], []]
            if not os.path.isdir(path_to_look):
                raise Exception("Couldn't find specified source "+path_to_look)
            for fname in os.listdir(path_to_look):
              exercise_id = fname.split('_')[-2]
              exercise_id_int = int(exercise_id[-1])-1
              tlen[exercise_id_int].append(np.load(os.path.join(path_to_look, fname), mmap_mode='r').shape[0])
            source_mean_len_dict[source_name] = [[np.mean(i), np.std(i), np.median(i)] for i in tlen]
        return source_mean_len_dict

# ds = Data_Stats(root_path='D:/MS-Project/raw_data')
# stats_dict = ds.get_stats_frame_len(source_select_list=['E', 'NE', 'BP', 'P', 'S'])
# ds.print_stats(stats_dict)