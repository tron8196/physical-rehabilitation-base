import copy
from BaselineModelFactory import *

import wandb

from neurorehabilitation.comparitive_study import BaselineModelFactory
from neurorehabilitation.data.datasets.kimore.DataLoaderKIMORE import load_raw_data_with_labels
from neurorehabilitation.data.datasets.uiprmd.UIPRMDResults import UIPRMDResults
from neurorehabilitation.data.datasets.uiprmd.data_loaders.ViconDataLoader import load_raw_data_with_scores
from neurorehabilitation.transformer.EmbeddingModelFactory import EmbeddingModels
from neurorehabilitation.transformer.TransformerFactory import TransformerFactory
import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback
from sklearn.metrics import mean_squared_error
import datetime

##########################################################################
def train_test_split(arr, label, test_size):
    arr_len = arr.shape[0]
    train_idx = np.random.choice(np.arange(arr_len), int(arr_len*(1-test_size)), replace=False)
    test_idx = np.setdiff1d(np.arange(arr_len), train_idx)
    arr_copy = copy.deepcopy(arr)
    label_copy = copy.deepcopy(label)
    return arr_copy[train_idx], arr_copy[test_idx], label_copy[train_idx], label_copy[test_idx]

###########################################################################
# Code to re-order the 117 dimensional skeleton data from the Vicon optical tracker into trunk, left arm, right arm, left leg and right leg
def reorder_data(x):
    X_trunk = np.zeros((x.shape[0],x.shape[1],12))
    X_left_arm = np.zeros((x.shape[0],x.shape[1],18))
    X_right_arm = np.zeros((x.shape[0],x.shape[1],18))
    X_left_leg = np.zeros((x.shape[0],x.shape[1],21))
    X_right_leg = np.zeros((x.shape[0],x.shape[1],21))
    X_trunk =  np.concatenate((x[:,:,15:18], x[:,:,18:21], x[:,:,24:27], x[:,:,27:30]), axis = 2)
    X_left_arm = np.concatenate((x[:,:,81:84], x[:,:,87:90], x[:,:,93:96], x[:,:,99:102], x[:,:,105:108], x[:,:,111:114]), axis = 2)
    X_right_arm = np.concatenate((x[:,:,84:87], x[:,:,90:93], x[:,:,96:99], x[:,:,102:105], x[:,:,108:111], x[:,:,114:117]), axis = 2)
    X_left_leg = np.concatenate((x[:,:,33:36], x[:,:,39:42], x[:,:,45:48], x[:,:,51:54], x[:,:,57:60], x[:,:,63:66], x[:,:,69:72]), axis = 2)
    X_right_leg = np.concatenate((x[:,:,36:39], x[:,:,42:45], x[:,:,48:51], x[:,:,54:57], x[:,:,60:63], x[:,:,66:69], x[:,:,72:75]), axis = 2)
    x_segmented = np.concatenate((X_trunk, X_right_arm, X_left_arm, X_right_leg, X_left_leg),axis = -1)
    return x_segmented

# Code to re-order the 88 dimensional skeleton data from Kinect into trunk, left arm, right arm, left leg and right leg
def reorder_data_kinect(x):
    X_trunk = np.zeros((x.shape[0],x.shape[1],16))
    X_left_arm = np.zeros((x.shape[0],x.shape[1],16))
    X_right_arm = np.zeros((x.shape[0],x.shape[1],16))
    X_left_leg = np.zeros((x.shape[0],x.shape[1],16))
    X_right_leg = np.zeros((x.shape[0],x.shape[1],16))
    X_trunk =  x[:,:,0:16]
    X_left_arm = x[:,:,16:32]
    X_right_arm = x[:,:,32:48]
    X_left_leg = x[:,:,48:64]
    X_right_leg = x[:,:,64:80]
    x_segmented = np.concatenate((X_trunk, X_right_arm, X_left_arm, X_right_leg, X_left_leg),axis = -1)
    return x_segmented




results_list = []
results = UIPRMDResults(experiment_type='CNN_Baseline_KIMORE')

# for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     exercise_id = 'e' + '0'*(2 - len(str(i))) + str(i)

for i in [5]:
    exercise_id = 'Es' + str(i)
    # exercise_id = 'e01'
    print(exercise_id)

    ###########################################################################
    '''
    The original baseline paper did not have splits for test and validation, so without the guidance in the earlystopping,
    it is not possible to be sure of the best performing model, hence we choose to give 10.5% to test, 19.5% to validation 
    and rest 70% for training of the model.
    '''
    # Split the vicon_data into training and validation sets
    # Training set: 70%
    # Validation set: 20%
    # Test set: 10%
    # correct_input, correct_label, incorrect_input, incorrect_label = load_raw_data_with_scores(exercise_id=exercise_id)
    # combined_input = np.vstack((correct_input, incorrect_input))
    # combined_input = reorder_data(combined_input)
    # combined_label = np.vstack((correct_label, incorrect_label))

    #KIMORE
    combined_input, combined_label = load_raw_data_with_labels(exercise_id=exercise_id)
    combined_input = reorder_data_kinect(combined_input)
    print(combined_input.shape, combined_label.shape)



    x_train, x_valid, y_train, y_valid = train_test_split(combined_input, combined_label, test_size=0.3)
    ##########################################################################################
    REPEAT_ITER = 5
    MAE_list = []
    RMSE_list = []

    _, timesteps, n_dim = combined_input.shape

    for i in range(REPEAT_ITER):
        baseline_model = deep_cnn_network(timesteps, n_dim)
        # transformer.summary()
        baseline_model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4)
        )


        # callbacks = [tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
        #              WandbCallback(), tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)]

        now = datetime.datetime.now
        t = now()
        history = baseline_model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=10000,
        batch_size=10,
        callbacks=callbacks,
        verbose=1
        )


        # Plot the prediction of the model for the training and validation sets
        pred_train = baseline_model.predict(x_train)

        pred_test = baseline_model.predict(x_valid)

        # Print the minimum loss
        print("Training loss", np.min(history.history['loss']))
        print("Validation loss", np.min(history.history['val_loss']))


        print('Training time: %s' % (now() - t))
        pred_test = baseline_model.predict(x_valid)
        print('The results for exercise '+exercise_id)
        print()
        # Calculate the cumulative deviation and rms deviation for the validation set
        test_dev = np.abs(np.squeeze(pred_test) - y_valid)
        # Cumulative deviation
        mean_abs_dev = np.mean(test_dev)
        # RMS deviation
        # rms_dev = mean_squared_error(pred_test, y_test)
        rms_dev = np.sqrt(mean_squared_error(pred_test, y_valid))
        print('Mean absolute deviation:', mean_abs_dev)
        print('RMSE deviation:', rms_dev)
        baseline_model = None

        MAE_list.append(mean_abs_dev)
        RMSE_list.append(rms_dev)
        results.results_dict[exercise_id].append((mean_abs_dev, rms_dev))


    mae_mean = np.mean(MAE_list)
    rmse_mean = np.mean(RMSE_list)
    print('Average MAE and RMSE')
    results_list.append((mae_mean, rmse_mean, exercise_id))
    print(mae_mean, rmse_mean)
    results.write_results_exercise(exercise_id=exercise_id)

print()
print('###################################################')
for mae_mean, rmse_mean, exercise_id in results_list:
    print(mae_mean, rmse_mean, exercise_id)
    print('------------------------------------')
print('###################################################')