import numpy as np
from scipy.spatial.transform import Rotation
import scipy.interpolate as interp

class DataProcessing:
    def __init__(self):
        pass
    '''
    This function expects a NxD dimensional array, every 4 elements in each row represent a quaternion. The function
    converts this to a euler angle format, this functions expects the kinect 25joint location format
    '''
    def convert_quaternion_to_euler_angle(self, arr):
        n_frames, n_dim = arr.shape
        rot = Rotation.from_quat(arr.reshape(-1, 4))
        euler = rot.as_euler('xyz', degrees=True)
        return euler.reshape(n_frames, -1)

    def get_velocity_vec(self, arr):
        n_frames, n_dim = arr.shape
        vel = np.power(np.diff(arr.reshape(-1, 3), axis=0), 2)

    '''
    This function should return magnitude of the velocity.
    input :- N*75(vx1, vy1, vz1, vx2, vy2, vz2, ....vz75)
    output :- N*25(v1, v2, ..., v25)
    '''
    def get_velocity_mag(self, arr):
        pass


    '''
    This takes a 2D array as input and interpolates each column to the specified interp_to_frame_len
    input :- NxD, interp_to_frame_len = K
    output :- KxD
    '''

    def interpolate(self, arr, interp_to_frame_len, method='linear'):
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