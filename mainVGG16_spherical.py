################################################################################
#
#   Author: Christopher Angelini
#
#   Porpoise: Main file for the VGG16_SPHERICAL architecture
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

        self.time_step = 1
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

    # Read in the image for size
    image = cv2.imread(full_train_images[0, 0], 1)

    # Create network
    # Grab the base model from keras
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

    # Grab the feature representation from the prebuilt model
    x = base_model.output
    # Randomly added layer
    x = layers.GlobalMaxPooling2D()(x)
    # Fully connected classification layer
    preds = layers.Dense(syn_head_gen.num_bins, activation='softmax')(x)
    # Recreate the model with the changed output layer
    model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)

    # Print model layers
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    # Define trainable layers
    for layer in model.layers:
        layer.trainable = False
    # or if we want to set the last 8 layers of the network to be trainable
    for layer in model.layers[-8:]:
        layer.trainable = True
        print(layer.name)

    #model = load_model('./resnetModels/042619_055142/wholeModel.h5')
    params = {'batch_size': args.batch_size,
              'time_steps': args.time_step,
              'input_shape': (64,64,3),
              'num_classes': syn_head_gen.num_bins,
              'shuffle': True}
    # Call sequence generator for training
    training_generator = SynHeadSequencer(full_train_images, full_train_bin, full_train_bg, full_train_video_start, **params)
    # Call sequence generator for testing
    testing_generator = SynHeadSequencer(test_images, test_bin, test_bg, test_video_start, **params)
    # Call optimizer (SGD, ADAM, NADAM)
    opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.8, beta_2=0.888, epsilon=None, decay=0.0, amsgrad=True)
    #opt = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    # Compile the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # log epoch data
    csv_logger = CSVLogger(args.outputPath+'trainMctrainlog.csv')
    # Fit model
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(training_generator),
                        epochs=25, shuffle=True, workers=4,  use_multiprocessing=False,
                        validation_data=testing_generator, validation_steps=len(testing_generator),
                        callbacks=[csv_logger])

    model.save(args.outputPath + 'wholeModel.h5')
    model.save_weights(args.outputPath + 'weightsOnlyModel.h5')
