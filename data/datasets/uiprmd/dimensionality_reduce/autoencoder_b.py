from neurorehabilitation.data.datasets.uiprmd.data_loaders.ViconDataLoader import load_raw_data


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Conv1D, MaxPool1D, Dropout, Flatten, Dense, Conv1DTranspose, Concatenate, UpSampling1D, GlobalAveragePooling1D, Lambda, GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow.python.keras.layers import CuDNNLSTM
import time
import numpy as np
import random

###########################################################################

start = time.time()
###########################################################################
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

###########################################################################

batch_size = 10
exercise_id = "e04"
RAW_DATA_BASE_DIR_PATH = "../extracted_raw_data"
X_correct, X_incorrect = load_raw_data(exercise_id=exercise_id,
                                       RAW_DATA_BASE_DIR_PATH=RAW_DATA_BASE_DIR_PATH)
print(X_correct)
num_examples, n_timesteps, n_dim = X_correct.shape

data_correct = np.zeros((X_correct.shape[0],n_timesteps+100,n_dim))
for i in range(X_correct.shape[0]):
    data_correct[i,:,:] = np.concatenate((np.concatenate((np.tile(X_correct[i,0,:],[50, 1]), X_correct[i,:,:])), np.tile(X_correct[i,-1,:],[50, 1])))

data_incorrect = np.zeros((X_incorrect.shape[0],n_timesteps+100,n_dim))
for i in range(X_incorrect.shape[0]):
    data_incorrect[i,:,:] = np.concatenate((np.concatenate((np.tile(X_incorrect[i,0,:],[50, 1]), X_incorrect[i,:,:])), np.tile(X_incorrect[i,-1,:],[50, 1])))

input_seq = Input(shape=(n_timesteps+100,n_dim))
encoded1 = CuDNNLSTM(30,return_sequences = True)(input_seq)
encoded2 = CuDNNLSTM(10,return_sequences = True)(encoded1)
# Encoded representation of the input, 340x4 vector
encoded = CuDNNLSTM(4,return_sequences = True)(encoded2)
# Decoder layers
decoded1 = CuDNNLSTM(10,return_sequences = True)(encoded)
decoded2 = CuDNNLSTM(30,return_sequences = True)(decoded1)
decoded = CuDNNLSTM(n_dim, return_sequences = True)(decoded2)

# The model maps an input to its reconstruction
autoencoder = Model(inputs=input_seq, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


trainidx = random.sample(range(0,data_correct.shape[0]),int(data_correct.shape[0]*0.7))
valididx = np.setdiff1d(np.arange(0,data_correct.shape[0],1),trainidx)
train_data = data_correct[trainidx,:,:]
valid_data = data_correct[valididx,:,:]




# Train an autoencoder on the correct data sequences


# Request to stop before reaching the number of epochs if the validation loss does not decrease for 1000 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 1000)

history = autoencoder.fit(train_data, train_data, epochs = 10000, batch_size = batch_size, shuffle=True,
                validation_data=(valid_data, valid_data), verbose = 1, callbacks = [early_stopping])

# end time
end = time.time()

# total time taken
print(f"Runtime of the model is {(end - start)/3600}")

# Create an encoder model, that maps an input to its encoded representation
encoder = Model(inputs=input_seq, outputs=encoded)

# Test the encoder model
encoded_seqs = encoder.predict(data_correct)


# Remove the added first and last 50 frames
encoded_seqs = encoded_seqs[:,50:-50,:]
'../dim_reduce_data/'
print(encoded_seqs.shape, 'encoded sequences shape')
# Reshape the encoded sequences, because savetxt saves two dimensional data
seqs = encoded_seqs.reshape(encoded_seqs.shape[0],encoded_seqs.shape[1]*encoded_seqs.shape[2])
print(seqs.shape, 'encoded sequences shape for saving')
# Save the data in the file 'Autoencoder_Output_Correct.csv'
np.savetxt("../dim_reduce_data/Autoencoder_Output_Correct_"+exercise_id+".csv", seqs, fmt='%.5f',delimiter=',')

# Reduce the dimensionality of the incorrect sequences
encoded_seqs_incorrect = encoder.predict(data_incorrect)


# Remove the added first and last 50 frames
encoded_seqs_incorrect = encoded_seqs_incorrect[:,50:-50,:]

print(encoded_seqs_incorrect.shape, 'encoded incorrect sequences shape')
# Reshape the encoded sequences, because savetxt saves only tow dimensional data
seqs_incorrect = encoded_seqs_incorrect.reshape(encoded_seqs_incorrect.shape[0],encoded_seqs_incorrect.shape[1]*encoded_seqs_incorrect.shape[2])
print(seqs_incorrect.shape, 'encoded incorrect sequences shape for saving')
# Save the incorrect data in the file 'Autoencoder_Output_Incorrect.csv'
np.savetxt("../dim_reduce_data/Autoencoder_Output_Incorrect_"+exercise_id+".csv", seqs_incorrect, fmt='%.5f',delimiter=',')

ts = str(int(time.time()))
encoder.save('./model_savepoint/model_encoder_'+exercise_id+"_"+ts+'.h5')

