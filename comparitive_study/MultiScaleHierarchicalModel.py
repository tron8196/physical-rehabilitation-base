from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv1D, LSTM, Concatenate, Input, Flatten, Dense, Dropout, Lambda, UpSampling1D
from tensorflow.python.keras.layers import CuDNNLSTM
import datetime

now = datetime.datetime.now

class MultiScaleHierarchicalModel:
    def __init__(self):
        pass

    # Define a multibranch convolutional Inception-like block
    def MultiBranchConv1D(self, input, filters1, kernel_size1, strides1, strides2):
        x1 = Conv1D(filters=filters1, kernel_size=kernel_size1+2, strides=strides1, padding='same', activation='relu')(input)
        x1 = Dropout(0.25)(x1)
        x2 = Conv1D(filters=filters1, kernel_size=kernel_size1+6, strides=strides1, padding='same', activation='relu')(input)
        x2 = Dropout(0.25)(x2)
        x3 = Conv1D(filters=filters1, kernel_size=kernel_size1+12, strides=strides1, padding='same', activation='relu')(input)
        x3 = Dropout(0.25)(x3)
        y1 = Concatenate(axis=-1)([x1, x2, x3])

        x4 = Conv1D(filters=filters1, kernel_size=kernel_size1, strides=strides2, padding='same', activation='relu')(y1)
        x4 = Dropout(0.25)(x4)
        x5 = Conv1D(filters=filters1, kernel_size=kernel_size1+2, strides=strides2, padding='same', activation='relu')(y1)
        x5 = Dropout(0.25)(x5)
        x6 = Conv1D(filters=filters1, kernel_size=kernel_size1+4, strides=strides2, padding='same', activation='relu')(y1)
        x6 = Dropout(0.25)(x6)
        x = Concatenate(axis=-1)([x4, x5, x6])
        return x


    # Define a temporal pyramid network
    def TempPyramid(self, input_f, input_2, input_4, input_8, seq_len, n_dims):
        #### Full scale sequences
        conv1 = self.MultiBranchConv1D(input_f, 64, 3, 2, 2)

        #### Half scale sequences
        conv2 = self.MultiBranchConv1D(input_2, 64, 3, 2, 1)

        #### Quarter scale sequences
        conv3 = self.MultiBranchConv1D(input_4, 64, 3, 1, 1)

        #### Eighth scale sequences
        conv4 = self.MultiBranchConv1D(input_8, 64, 3, 1, 1)
        upsample1 = UpSampling1D(size=2)(conv4)

        #### Recurrent layers
        x = Concatenate(axis=-1)([conv1, conv2, conv3, upsample1])
        return x

    def build_model(self, inp_shape):
        timesteps, n_dim = inp_shape
        n_dim = 90  # dimension after segmenting the data into body parts
        n_dim1 = 12  # trunk dimension
        n_dim2 = 18  # arms dimension
        n_dim3 = 21  # legs dimension

        #### Full scale sequences
        seq_input = Input(shape=(timesteps, n_dim), name='full_scale')

        seq_input_trunk = Lambda(lambda x: x[:, :, 0:12])(seq_input)
        seq_input_left_arm = Lambda(lambda x: x[:, :, 12:30])(seq_input)
        seq_input_right_arm = Lambda(lambda x: x[:, :, 30:48])(seq_input)
        seq_input_left_leg = Lambda(lambda x: x[:, :, 48:69])(seq_input)
        seq_input_right_leg = Lambda(lambda x: x[:, :, 69:90])(seq_input)

        #### Half scale sequences
        seq_input_2 = Input(shape=(int(timesteps / 2), n_dim), name='half_scale')

        seq_input_trunk_2 = Lambda(lambda x: x[:, :, 0:12])(seq_input_2)
        seq_input_left_arm_2 = Lambda(lambda x: x[:, :, 12:30])(seq_input_2)
        seq_input_right_arm_2 = Lambda(lambda x: x[:, :, 30:48])(seq_input_2)
        seq_input_left_leg_2 = Lambda(lambda x: x[:, :, 48:69])(seq_input_2)
        seq_input_right_leg_2 = Lambda(lambda x: x[:, :, 69:90])(seq_input_2)

        #### Quarter scale sequences
        seq_input_4 = Input(shape=(int(timesteps / 4), n_dim), name='quarter_scale')

        seq_input_trunk_4 = Lambda(lambda x: x[:, :, 0:12])(seq_input_4)
        seq_input_left_arm_4 = Lambda(lambda x: x[:, :, 12:30])(seq_input_4)
        seq_input_right_arm_4 = Lambda(lambda x: x[:, :, 30:48])(seq_input_4)
        seq_input_left_leg_4 = Lambda(lambda x: x[:, :, 48:69])(seq_input_4)
        seq_input_right_leg_4 = Lambda(lambda x: x[:, :, 69:90])(seq_input_4)

        #### Eighth scale sequences
        seq_input_8 = Input(shape=(int(timesteps / 8), n_dim), name='eighth_scale')

        seq_input_trunk_8 = Lambda(lambda x: x[:, :, 0:12])(seq_input_8)
        seq_input_left_arm_8 = Lambda(lambda x: x[:, :, 12:30])(seq_input_8)
        seq_input_right_arm_8 = Lambda(lambda x: x[:, :, 30:48])(seq_input_8)
        seq_input_left_leg_8 = Lambda(lambda x: x[:, :, 48:69])(seq_input_8)
        seq_input_right_leg_8 = Lambda(lambda x: x[:, :, 69:90])(seq_input_8)

        concat_trunk = self.TempPyramid(seq_input_trunk, seq_input_trunk_2, seq_input_trunk_4, seq_input_trunk_8, timesteps,
                                   n_dim1)
        concat_left_arm = self.TempPyramid(seq_input_left_arm, seq_input_left_arm_2, seq_input_left_arm_4, seq_input_left_arm_8,
                                      timesteps, n_dim2)
        concat_right_arm = self.TempPyramid(seq_input_right_arm, seq_input_right_arm_2, seq_input_right_arm_4,
                                       seq_input_right_arm_8, timesteps, n_dim2)
        concat_left_leg = self.TempPyramid(seq_input_left_leg, seq_input_left_leg_2, seq_input_left_leg_4, seq_input_left_leg_8,
                                      timesteps, n_dim3)
        concat_right_leg = self.TempPyramid(seq_input_right_leg, seq_input_right_leg_2, seq_input_right_leg_4,
                                       seq_input_right_leg_8, timesteps, n_dim3)
        concat = Concatenate(axis=-1)([concat_trunk, concat_left_arm, concat_right_arm, concat_left_leg, concat_right_leg])
        rec = CuDNNLSTM(80, return_sequences=True)(concat)
        rec1 = CuDNNLSTM(40, return_sequences=True)(rec)
        rec1 = CuDNNLSTM(40, return_sequences=True)(rec1)
        rec2 = CuDNNLSTM(80)(rec1)
        out = Dense(1, activation='sigmoid')(rec2)
        model = Model(inputs=[seq_input, seq_input_2, seq_input_4, seq_input_8], outputs=out)
        return model


    def build_model_kinect(self, inp_shape):
        timesteps, n_dim = inp_shape
        n_dim = 80  # dimension after segmenting the data into body parts
        n_dim1 = 16

        #### Full scale sequences
        seq_input = Input(shape=(timesteps, n_dim), name='full_scale')

        seq_input_trunk = Lambda(lambda x: x[:, :, 0:16])(seq_input)
        seq_input_left_arm = Lambda(lambda x: x[:, :, 16:32])(seq_input)
        seq_input_right_arm = Lambda(lambda x: x[:, :, 32:48])(seq_input)
        seq_input_left_leg = Lambda(lambda x: x[:, :, 48:64])(seq_input)
        seq_input_right_leg = Lambda(lambda x: x[:, :, 64:80])(seq_input)

        #### Half scale sequences
        seq_input_2 = Input(shape=(int(timesteps / 2), n_dim), name='half_scale')


        seq_input_trunk_2 = Lambda(lambda x: x[:, :, 0:16])(seq_input_2)
        seq_input_left_arm_2 = Lambda(lambda x: x[:, :, 16:32])(seq_input_2)
        seq_input_right_arm_2 = Lambda(lambda x: x[:, :, 32:48])(seq_input_2)
        seq_input_left_leg_2 = Lambda(lambda x: x[:, :, 48:64])(seq_input_2)
        seq_input_right_leg_2 = Lambda(lambda x: x[:, :, 64:80])(seq_input_2)



        #### Quarter scale sequences
        seq_input_4 = Input(shape=(int(timesteps / 4), n_dim), name='quarter_scale')

        seq_input_trunk_4 = Lambda(lambda x: x[:, :, 0:16])(seq_input_4)
        seq_input_left_arm_4 = Lambda(lambda x: x[:, :, 16:32])(seq_input_4)
        seq_input_right_arm_4 = Lambda(lambda x: x[:, :, 32:48])(seq_input_4)
        seq_input_left_leg_4 = Lambda(lambda x: x[:, :, 48:64])(seq_input_4)
        seq_input_right_leg_4 = Lambda(lambda x: x[:, :, 64:80])(seq_input_4)

        #### Eighth scale sequences
        seq_input_8 = Input(shape=(int(timesteps / 8), n_dim), name='eighth_scale')

        seq_input_trunk_8 = Lambda(lambda x: x[:, :, 0:16])(seq_input_8)
        seq_input_left_arm_8 = Lambda(lambda x: x[:, :, 16:32])(seq_input_8)
        seq_input_right_arm_8 = Lambda(lambda x: x[:, :, 32:48])(seq_input_8)
        seq_input_left_leg_8 = Lambda(lambda x: x[:, :, 48:64])(seq_input_8)
        seq_input_right_leg_8 = Lambda(lambda x: x[:, :, 64:80])(seq_input_8)

        concat_trunk = self.TempPyramid(seq_input_trunk, seq_input_trunk_2, seq_input_trunk_4, seq_input_trunk_8, timesteps,
                                   n_dim1)
        concat_left_arm = self.TempPyramid(seq_input_left_arm, seq_input_left_arm_2, seq_input_left_arm_4, seq_input_left_arm_8,
                                      timesteps, n_dim1)
        concat_right_arm = self.TempPyramid(seq_input_right_arm, seq_input_right_arm_2, seq_input_right_arm_4,
                                       seq_input_right_arm_8, timesteps, n_dim1)
        concat_left_leg = self.TempPyramid(seq_input_left_leg, seq_input_left_leg_2, seq_input_left_leg_4, seq_input_left_leg_8,
                                      timesteps, n_dim1)
        concat_right_leg = self.TempPyramid(seq_input_right_leg, seq_input_right_leg_2, seq_input_right_leg_4,
                                       seq_input_right_leg_8, timesteps, n_dim1)
        concat = Concatenate(axis=-1)([concat_trunk, concat_left_arm, concat_right_arm, concat_left_leg, concat_right_leg])
        rec = CuDNNLSTM(80, return_sequences=True)(concat)
        rec1 = CuDNNLSTM(40, return_sequences=True)(rec)
        rec1 = CuDNNLSTM(40, return_sequences=True)(rec1)
        rec2 = CuDNNLSTM(80)(rec1)
        out = Dense(1, activation='sigmoid')(rec2)
        model = Model(inputs=[seq_input, seq_input_2, seq_input_4, seq_input_8], outputs=out)
        return model
