import numpy as np
from matplotlib import pyplot as plt


arr = np.load('./correct_movement_e01_arr.npy')

arr_left_elbow = arr[:, :, 13*3:13*3+3]
arr_left_elbow = arr_left_elbow.max(axis=0)
plt.plot(arr_left_elbow)
plt.show()