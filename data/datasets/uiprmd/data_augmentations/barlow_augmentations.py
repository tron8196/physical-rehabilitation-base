import numpy as np

import itertools
import scipy.interpolate as interp

#this is designed for the Vicon system.
MAX_JOINT_LANDMARKS = 39
'''
1. Load the combined_array_proper_movements.npy, this contains vicon_data for all exercises 
   with all subjects and all repetition episodes.

2. any self-supervised algorithm specially lay stress on vicon_data augmentation techniques, for my vicon_data, I believe the following 
vicon_data augmentation techniques can work.
    a. reversal
    b. random patches, however instead of putting patches on all the dimensions of the feature vector, the process can 
       be thought of as an occlusion event, thus hiding randomly selected joints for a few frames should make more sense.
    c. up and down-sampling can mimic the action being performed quickly or slowly, which still is the same action.
    d. low-grade noise which doesn't distort the joint structure of the body can be explored
       (look at the ERD model for de-noising of kinect vicon_data).
'''


# below is a section of vicon_data augmentation functions which will be used during the training.

def interpolate(arr, interp_to_frame_len, method='linear'):
    n_frames, n_dim = arr.shape
    interp_arr = np.zeros((interp_to_frame_len, arr.shape[-1]))
    if method == 'linear':
        for i in range(n_dim):
            arr_interp = interp.interp1d(np.arange(arr.shape[0]), arr[:, i])
            interp_arr[:, i] = arr_interp(np.linspace(0, arr.shape[0] - 1, interp_to_frame_len))
        return interp_arr
    elif method == 'cubic':
        for i in range(n_dim):
            arr_interp = interp.interp1d(np.arange(arr.shape[0]), arr[:, i], kind='cubic')
            interp_arr[:, i] = arr_interp(np.linspace(0, arr.shape[0] - 1, interp_to_frame_len))
        return interp_arr
    elif method == 'nearest':
        pass


def random_reversal(arr, reversal_prob=0.2):
    copy_arr = np.copy(arr)
    for batch_index in range(copy_arr.shape[0]):
        if np.random.rand() <= reversal_prob:
            copy_arr[batch_index] = copy_arr[batch_index][::-1]
    return copy_arr

'''
1. randomly select N joints(maxed at max_joints_to_occlude_at_step) at every modulo (occlude_for_frames), and make them zero.
2. the joints are occluded for occlude_for_frames parameter.

'''

def random_occlusions(arr, occlude_for_frames=10, max_joints_to_occlude_at_step=10, max_occlusions=10):
    batch_size, n_timesteps, n_features = arr.shape
    #this gives the available steps, which can be selected for occlusion.
    n_steps = n_timesteps // occlude_for_frames
    copy_arr = np.copy(arr)
    for batch_index in range(copy_arr.shape[0]):
        steps_to_occlude = np.random.choice(range(n_steps), (max_occlusions,), replace=False)
        masking_arr = np.ones((copy_arr[batch_index].shape[0], copy_arr[batch_index].shape[1]), dtype=arr.dtype)
        for curr_timestep in range(n_timesteps):
            if curr_timestep//occlude_for_frames in steps_to_occlude:
                joints_to_occlude = np.random.randint(1, max_joints_to_occlude_at_step+1)
                #this will give the joint number to mask
                select_joints_to_occlude = np.random.choice(range(MAX_JOINT_LANDMARKS), joints_to_occlude, replace=False)
                joint_index_to_mask = [[3*i, 3*i+1, 3*i+2] for i in select_joints_to_occlude.tolist()]
                merged = list(itertools.chain(*joint_index_to_mask))
                masking_arr[curr_timestep][merged] = 0
        copy_arr[batch_index] = np.multiply(copy_arr[batch_index], masking_arr)
    return copy_arr

'''
1. toss a fair coin and choose up or down-sampling
2. choose scale from (1, 1.25) for up and (0.75, 1) for down-sampling. randomly select the scaling factor according
   to the coin toss
'''

def random_up_down_sample(arr, max_interp_factor=0.25):
    batch_size, n_timesteps, n_features = arr.shape
    copy_arr = np.copy(arr)
    for batch_index in range(copy_arr.shape[0]):
        # up-sampling
        if np.random.rand() > 0.5:
            new_sample_len = int(n_timesteps * np.random.randint(100, 125+1)/100)
            copy_arr[batch_index] = interpolate(copy_arr[batch_index], interp_to_frame_len=new_sample_len)[:n_timesteps]
        else:
            new_sample_len = int(n_timesteps * np.random.randint(75, 100+1)/100)
            interp_arr = interpolate(copy_arr[batch_index], interp_to_frame_len=new_sample_len)
            padding_arr = np.concatenate((interp_arr, np.zeros((n_timesteps - new_sample_len, n_features))), axis=0)
            copy_arr[batch_index] = padding_arr
    return copy_arr


def random_masking(arr, occlude_for_frames=10, max_occlusions=10):
    batch_size, n_timesteps, n_features = arr.shape
    #this gives the available steps, which can be selected for occlusion.
    n_steps = n_timesteps // occlude_for_frames
    copy_arr = np.copy(arr)
    for batch_index in range(copy_arr.shape[0]):
        steps_to_occlude = np.random.choice(range(n_steps), (max_occlusions,), replace=False)
        masking_arr = np.ones((copy_arr[batch_index].shape[0], copy_arr[batch_index].shape[1]), dtype=arr.dtype)
        for curr_timestep in range(n_timesteps):
            if curr_timestep//occlude_for_frames in steps_to_occlude:
                masking_arr[curr_timestep: curr_timestep+occlude_for_frames] = 0
        copy_arr[batch_index] = np.multiply(copy_arr[batch_index], masking_arr)
    return copy_arr



# a = np.random.randn(16, 240, 117)
# print('shape before', a.shape)
# a = random_up_down_sample(a)
# print('shape after', a.shape)
#
