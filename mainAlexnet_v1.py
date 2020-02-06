import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import os
import tensorflow.keras.layers as layers
import cv2
from tensorflow.keras.models import load_model

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

        self.time_step = 1
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

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=4, padding='same', activation=tf.nn.relu,
                            input_shape=(64, 64, 3)))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
    model.add(layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(syn_head_gen.num_bins, activation='softmax'))

    #model = load_model('./resnetModels/042619_055142/wholeModel.h5')
    params = {'batch_size': args.batch_size,
              'time_steps': args.time_step,
              'input_shape': (64,64,3),
              'num_classes': syn_head_gen.num_bins,
              'shuffle': True}

    training_generator = SynHeadSequencer(full_train_images, full_train_yaw, full_train_bg, full_train_video_start, **params)

    testing_generator = SynHeadSequencer(test_images, test_yaw, test_bg, test_video_start, **params)

    opt = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(training_generator),
                        epochs=20, shuffle=True, workers=2,  use_multiprocessing=True,
                        validation_data=testing_generator, validation_steps=len(testing_generator))

    model.save(args.outputPath + 'wholeModel.h5')
    model.save_weights(args.outputPath + 'weightsOnlyModel.h5')