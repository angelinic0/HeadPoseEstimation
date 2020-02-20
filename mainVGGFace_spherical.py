################################################################################
#
#   Author: Christopher Angelini
#
#   Porpoise: Main file for the VGG16_SPHERICAL (with the model fully trainable) architecture
#
################################################################################
import os
from time import gmtime, strftime

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from dataset import SynHeadSpherical
from sequencer import SynHeadSequencer

np.random.seed(42)

# Arguement class
class Args:
    def __init__(self):
        # Directories from each of the separate datasets
        self.brein_train_dir = '../data/synhead2_release/breitenstein/'
        self.kinetic_train_dir = '../data/synhead2_release/kinect/'
        self.soft_train_dir = '../data/synhead2_release/softkinetic/'
        self.biwi_test_dir = '../data/synhead2_release/BIWI/'

        # Create a save path
        now = strftime("%m%d%y_%H%M%S", gmtime())
        self.outputPath = './resnetModels/' + now + '/'

        # If output directory doesn't exist then make the directory
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        # Create a training list of model numbers
        self.brein_model_train_list = [6, 3, 0, 9, 7, 5, 1, 8, 0, 6, 8, 7, 3, 5, 9, 1, 7, 5, 8, 6, 0, 1, 9, 3, 0, 8]
        self.kinetic_model_train_list = [7, 3, 5, 9, 1]
        self.soft_model_train_list = [0, 6, 8, 7, 3, 5, 9, 1, 7, 5, 8, 6, 3, 0, 8]
        # Create a model list for testing
        self.biwi_model_test_list = [2, 4, 4, 2, 4, 2]

        # Create a movement list
        self.brein_move_train_list = np.arange(len(self.brein_model_train_list)).tolist()
        self.kinetic_move_train_list = np.arange(len(self.kinetic_model_train_list)).tolist()
        self.soft_move_train_list = np.arange(len(self.soft_model_train_list)).tolist()
        self.biwi_move_test_list = np.arange(len(self.biwi_model_test_list)).tolist()

        self.time_step = 5
        self.epochs = 50
        self.batch_size = 10

# Main function
if __name__=='__main__':
    args = Args()

    # Instantiate Dataset
    syn_head_gen = SynHeadSpherical(args.time_step)

    # Create datalists for each dataset
    # breitenstein
    b_train_images, b_train_bin, b_train_bg, b_train_video_start = \
        syn_head_gen.create_data(args.brein_train_dir, args.brein_model_train_list, args.brein_move_train_list)
    # kinect
    k_train_images, k_train_bin, k_train_bg, k_train_video_start = \
        syn_head_gen.create_data(args.kinetic_train_dir, args.kinetic_model_train_list, args.kinetic_move_train_list)
    # Soft kinect
    s_train_images, s_train_bin, s_train_bg, s_train_video_start = \
        syn_head_gen.create_data(args.soft_train_dir, args.soft_model_train_list, args.soft_move_train_list)

    # Determine video starting indices
    k_train_video_start = k_train_video_start + int(b_train_video_start[-1][0])
    s_train_video_start = s_train_video_start + int(k_train_video_start[-1][0])

    # Stack training images for image paths
    full_train_images = np.vstack(( b_train_images, k_train_images, s_train_images))

    # Append the training bins
    full_train_bin = np.append(np.append(b_train_bin, k_train_bin),s_train_bin)

    # Add image backgrounds
    full_train_bg = b_train_bg + k_train_bg + s_train_bg
    # Stack starting indices
    full_train_video_start = np.vstack((b_train_video_start, k_train_video_start, s_train_video_start))

    # Create data for testing images
    test_images, test_bin, test_bg, test_video_start = \
        syn_head_gen.create_data(args.biwi_test_dir, args.biwi_model_test_list, args.biwi_move_test_list)

    # Create network
    model = tf.keras.Sequential()
    model.add(layers.ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Convolution2D(4096, (7, 7), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(4096, (1, 1), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Convolution2D(2622, (1, 1)))

    model.add(layers.Flatten())

    input = layers.Input(batch_shape=(args.batch_size, args.time_step, 224, 224, 3))
    tdOut = td(model)(input)
    lstmOut = layers.LSTM(50, activation='tanh')(flOut)
    preds = layers.Dense(syn_head_gen.num_bins, activation='softmax')(lstmOut)

    tdmodel = tf.keras.models.Model(inputs=input, outputs=preds)

    #model = load_model('./resnetModels/042619_055142/wholeModel.h5')
    params = {'batch_size': args.batch_size,
              'time_steps': args.time_step,
              'input_shape': (224,224,3),
              'num_classes': syn_head_gen.num_bins,
              'shuffle': True}

    # Call sequence generator for training
    training_generator = SynHeadSequencer(full_train_images, full_train_bin, full_train_bg, full_train_video_start, **params)
    # Call sequence generator for training
    testing_generator = SynHeadSequencer(test_images, test_bin, test_bg, test_video_start, **params)
    # create optimizer
    opt = tf.keras.optimizers.Adam(lr=0.00001)
    # Compile the created model
    tdmodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # Fit the model
    tdmodel.fit_generator(generator=training_generator,
                        steps_per_epoch=len(training_generator),
                        epochs=20, shuffle=True, workers=4,  use_multiprocessing=False,
                        validation_data=testing_generator, validation_steps=len(testing_generator))
    # Save the model in an H5 file
    tdmodel.save(args.outputPath + 'wholeModel.h5')
    tdmodel.save_weights(args.outputPath + 'weightsOnlyModel.h5')
