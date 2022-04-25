import os
import numpy as np
import scipy
from scipy import interpolate as interp
from DataStats import DataStats
from DataProcessing import DataProcessing
from scipy.ndimage import gaussian_filter
'''
mean length is a hyper-parameter and has to be set by the user
important to note :- To allow for self-supervised learning all the exercises will have to be set to the same mean length
the code will be flexible enough for user to select which categories of subject and which exercises do they wan't in
the vicon_data
'''
SAVE_PATH_DIR = './processed_data'


class PrepareDataForNN:
    def __init__(self, mean_length=700, root_path='./raw_data'):
        self.mean_length = mean_length
        self.root_path = root_path
        self.Abbrevations = {'E':'Expert', 'NE':'NotExpert', 'BP':'BackPain', 'P':'Parkinson', 'S':'Stroke'}


    def prepare_data(self, source_select_list=['E', 'S'], exercise_select_list=['Es1'], interp_method='linear'):
        data_stats = DataStats(root_path=self.root_path)
        source_data_stats_dict = data_stats.get_stats_frame_len(source_select_list=source_select_list)
        data_stats.print_stats(source_data_stats_dict)
        data_processing = DataProcessing()
        combined_orient_arr = []
        combined_position_arr = []
        combined_labels = []
        for source in source_select_list:
            source_name = self.Abbrevations[source]
            source_stat_list = source_data_stats_dict[source_name]
            print(source_stat_list)
            path_to_look_pos = os.path.join(self.root_path, source_name, 'Position_Data')
            if not os.path.isdir(path_to_look_pos):
                raise Exception("Couldn't find specified source "+path_to_look_pos)
            #now get the same file with _orientation and _label from the neighbouring folders
            for fname in os.listdir(path_to_look_pos):
                try:
                    exercise_id = fname.split('_')[-2]
                    exercise_id_int = int(exercise_id[-1]) - 1
                except:
                    print(fname,exercise_id,exercise_id[-1])
                ex_mean = source_stat_list[exercise_id_int][0]
                ex_std_dev = source_stat_list[exercise_id_int][1]
                if exercise_id in exercise_select_list:
                    orient_fpath = os.path.join('../Orientation_Data', '_'.join(fname.split('_')[:-1])+'_oriention.npy')
                    label_fpath = os.path.join('../Label', '_'.join(fname.split('_')[:-2]) + '_Es1_label.npy')
                    pos_arr = np.load(os.path.join(path_to_look_pos, fname))
                    orient_arr = np.load(os.path.join(path_to_look_pos, orient_fpath))
                    label_arr = np.load(os.path.join(path_to_look_pos, label_fpath))
                    if  (pos_arr.shape[0] < ex_mean+3*ex_std_dev or pos_arr.shape[0] > ex_mean-3*ex_std_dev):
                        pos_arr_interp = data_processing.interpolate(pos_arr, self.mean_length, method=interp_method)
                        orient_arr_interp = data_processing.interpolate(orient_arr, self.mean_length, method=interp_method)
                        combined_orient_arr.append(orient_arr_interp)
                        combined_position_arr.append(pos_arr_interp)
                        combined_labels.append(label_arr[exercise_id_int])
                    else:
                        print('Dropping '+fname)
                        print(pos_arr.shape[0])
        combined_position_arr = np.stack(combined_position_arr, axis=0)
        combined_orient_arr = np.stack(combined_orient_arr, axis=0)
        combined_labels_arr = np.array(combined_labels)
        print(combined_position_arr.shape, combined_orient_arr.shape, combined_labels_arr.shape)
        return combined_position_arr, combined_orient_arr, combined_labels_arr

'''
give the root path for class constructor, in this dirs named as 'Stroke, Parkinson...' should be present
'''
data_nn = PrepareDataForNN(root_path='./raw_data')

exercise_id = 'Es5'

combined_position_arr, combined_orient_arr, combined_labels_arr = \
                    data_nn.prepare_data(source_select_list=['E', 'NE', 'S', 'BP', 'P'],
                                         exercise_select_list=[exercise_id])

#generate random index array
random_indexes = np.random.shuffle(np.arange(combined_orient_arr.shape[0]))

#randomize the array indexes
combined_position_arr = combined_position_arr[random_indexes]
combined_orient_arr = combined_orient_arr[random_indexes]
combined_labels_arr = combined_labels_arr[random_indexes]

#normalze the feature vector, axis=-1 is considered the feature dimension
# normalized_combined_orient_arr = (combined_orient_arr - combined_orient_arr.mean(axis=(-1), keepdims=True))/(1e-5+np.std(combined_orient_arr, axis=(-1), keepdims=True))
# normalized_combined_position_arr = (combined_position_arr - combined_position_arr.mean(axis=(-1), keepdims=True))/(1e-5+np.std(combined_position_arr, axis=(-1), keepdims=True))

eps = 1e-6
normalized_combined_orient_arr = np.squeeze(np.divide(combined_orient_arr - np.expand_dims(combined_orient_arr.mean(axis=1), axis=1),
                                 np.expand_dims(eps + np.nanstd(combined_orient_arr, axis=1), axis=1)))

normalized_combined_position_arr = np.squeeze(np.divide(combined_position_arr - np.expand_dims(combined_position_arr.mean(axis=1),
                                   axis=1), np.expand_dims(eps + np.nanstd(combined_position_arr, axis=1), axis=1)))
print(normalized_combined_orient_arr.shape)

n_examples, n_timesteps, n_dim = normalized_combined_orient_arr.shape
for example_id in range(n_examples):
    for feature_id in range(n_dim):
        dim_val_orient = normalized_combined_orient_arr[example_id, :, feature_id]
        dim_val_pos = normalized_combined_position_arr[example_id, :, feature_id]
        normalized_combined_orient_arr[example_id, :, feature_id] = gaussian_filter(dim_val_orient,
                                                                                    sigma=1.5)
        normalized_combined_position_arr[example_id, :, feature_id] = gaussian_filter(dim_val_pos,
                                                                                      sigma=1.5)


combined_labels_arr = np.round(combined_labels_arr/50, 4)

np.save(os.path.join(SAVE_PATH_DIR, 'extracted_pos_data_'+exercise_id), normalized_combined_position_arr)
np.save(os.path.join(SAVE_PATH_DIR, 'extracted_orient_data_'+exercise_id), normalized_combined_orient_arr)
np.save(os.path.join(SAVE_PATH_DIR, 'extracted_labels_data_'+exercise_id), combined_labels_arr)
