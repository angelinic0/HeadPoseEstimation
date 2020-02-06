import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import os
import tensorflow.keras.layers as layers
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TimeDistributed as td
from tensorflow.keras.callbacks import CSVLogger

from sequencer import SynHeadSequencer
from dataset import SynHead

np.random.seed(42)
tf.set_random_seed(3901)


class Args:
    def __init__(self):
        self.brein_train_dir = '../data/synhead2_release/breitenstein/'
        self.kinetic_train_dir = '../data/synhead2_release/kinect/'
        self.soft_train_dir = '../data/synhead2_release/softkinetic/'
        self.biwi_test_dir = '../data/synhead2_release/BIWI/'

        now = strftime("%m%d%y_%H%M%S", gmtime())
        self.outputPath = './resnetModels/' + now + '/'

        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        self.brein_model_train_list = [6, 3, 0, 9, 7, 5, 1, 8, 0, 6, 8, 7, 3, 5, 9, 1, 7, 5, 8, 6, 0, 1, 9, 3, 0, 8]
        self.kinetic_model_train_list = [7, 3, 5, 9, 1]
        self.soft_model_train_list = [0, 6, 8, 7, 3, 5, 9, 1, 7, 5, 8, 6, 3, 0, 8]
        self.biwi_model_test_list = [2, 4, 4, 2, 4, 2]

        self.brein_move_train_list = np.arange(len(self.brein_model_train_list)).tolist()
        self.kinetic_move_train_list = np.arange(len(self.kinetic_model_train_list)).tolist()
        self.soft_move_train_list = np.arange(len(self.soft_model_train_list)).tolist()
        self.biwi_move_test_list = np.arange(len(self.biwi_model_test_list)).tolist()

        self.time_step = 5
        self.epochs = 50
        self.batch_size = 10
        self.bin_size = 10
        self.max_angle = 90

if __name__=='__main__':
    args = Args()

    syn_head_gen = SynHead(args.time_step, args.bin_size, args.max_angle)

    b_train_images, b_train_pitch, b_train_yaw, b_train_roll, b_train_bg, b_train_video_start = \
        syn_head_gen.create_data(args.brein_train_dir, args.brein_model_train_list, args.brein_move_train_list)

    k_train_images, k_train_pitch, k_train_yaw, k_train_roll, k_train_bg, k_train_video_start = \
        syn_head_gen.create_data(args.kinetic_train_dir, args.kinetic_model_train_list, args.kinetic_move_train_list)

    s_train_images, s_train_pitch, s_train_yaw, s_train_roll, s_train_bg, s_train_video_start = \
        syn_head_gen.create_data(args.soft_train_dir, args.soft_model_train_list, args.soft_move_train_list)

    k_train_video_start = k_train_video_start + int(b_train_video_start[-1][0])
    s_train_video_start = s_train_video_start + int(k_train_video_start[-1][0])

    full_train_images = np.vstack(( b_train_images, k_train_images, s_train_images))

    full_train_pitch = np.append(np.append(b_train_pitch, k_train_pitch),s_train_pitch)
    full_train_yaw = np.append(np.append(b_train_yaw, k_train_yaw), s_train_yaw)
    full_train_roll = np.append(np.append(b_train_roll, k_train_roll), s_train_roll)

    full_train_bg = b_train_bg + k_train_bg + s_train_bg
    full_train_video_start = np.vstack((b_train_video_start, k_train_video_start, s_train_video_start))

    test_images, test_pitch, test_yaw, test_roll, test_bg, test_video_start = \
        syn_head_gen.create_data(args.biwi_test_dir, args.biwi_model_test_list, args.biwi_move_test_list)

    image = cv2.imread(full_train_images[0, 0], 1)

    base_model = tf.keras.applications.VGG16(weights='imagenet',
                                             input_shape=(64, 64, 3),
                                             include_top=False)
    cnnOut = base_model.layers[18].output

    cnnModel = tf.keras.Model(base_model.input, cnnOut)

    for layer in cnnModel.layers:
        layer.trainable = False
    # or if we want to set the first 20 layers of the network to be non-trainable
    for layer in cnnModel.layers[-8:]:
        layer.trainable = True

    input = layers.Input(batch_shape=(args.batch_size, args.time_step, 64,64,3))
    tdOut = td(cnnModel)(input)
    flOut = td(layers.Flatten())(tdOut)
    lstmOut = layers.LSTM(50, activation='tanh')(flOut)
    preds = layers.Dense(syn_head_gen.num_bins, activation='softmax')(lstmOut)

    model = tf.keras.models.Model(inputs=input, outputs=preds)

    #model = load_model('./resnetModels/042619_055142/wholeModel.h5')
    params = {'batch_size': args.batch_size,
              'time_steps': args.time_step,
              'input_shape': (64,64,3),
              'num_classes': syn_head_gen.num_bins,
              'shuffle': True}

    training_generator = SynHeadSequencer(full_train_images, full_train_roll, full_train_bg, full_train_video_start, **params)

    testing_generator = SynHeadSequencer(test_images, test_yaw, test_bg, test_video_start, **params)

    opt = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    csv_logger = CSVLogger(args.outputPath+'train.log')

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(training_generator),
                        epochs=20, shuffle=True, workers=2,  use_multiprocessing=False,
                        validation_data=testing_generator, validation_steps=len(testing_generator),
                        callbacks=[csv_logger])

    model.save(args.outputPath + 'wholeModel.h5')
    model.save_weights(args.outputPath + 'weightsOnlyModel.h5')