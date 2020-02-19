################################################################################
#
#   Author: Christopher Angelini
#
#   Porpoise: Traverses data library and loads image path information into a vector
#
################################################################################
import numpy as np
import glob
import os
import cv2

# Dataset information
################################################################################
#   This data folder has the following structure:
#   synhead2_release-|
#                    |-background-|
#                                 |-a01.png
#                                 |...
#                    |-BIWI-|
#                           |-00.csv (movement csvs)
#                           |-...
#                           |-23.csv
#                           |-f01-| (model folders)
#                           |-...-|
#                           |-m01-|
#                                 |-00 -| (movement folders per model)
#                                 |-...-|
#                                 |-23 -|
#                                       |-00.png (model pictures per movement )
#                    |-softkinect-| (same structure of BIWI)
#                    |-kinect-| (same structure of BIWI)
#                    |-breitenstein-| (same structure of BIWI)
################################################################################
data_sets = ['BIWI', 'breitenstein', 'kinect', 'softkinect']
models = ['f01', 'f02', 'f03', 'f04', 'f05', 'm01', 'm02', 'm03', 'm04', 'm05']
SYNHEAD_imW = 400
SYNHEAD_imH = 400

# Numpy random seed for constraining random variables
np.random.seed(42)

# Generate video frame list given a sub-dataset (data_dir), a 3D model number (model),
#   movement number (movement), and timestep (time_step)
def generate_video_frame_list(data_dir, model, movement, time_step):
    # Glob.glob breaks apart the concatinated string path to the .csv files
    tracks = glob.glob(os.path.join(data_dir, '*.csv'))
    # Sort the track files
    tracks.sort()

    # Load the delimited csv columns
    label = np.loadtxt(tracks[movement], delimiter=',')

    # Create an image path given data_dir, a model from the list of models, and
    #  a movement number
    img_path = os.path.join(data_dir, models[model], '%02d' % movement)
    # Break apart the file path
    img_files = glob.glob(os.path.join(img_path, '*.png'))
    # Sort
    img_files.sort()

    # Create an empty array for the inputs to the network, number of images minus
    # the number of images removed in the begining by the number of timesteps
    images = np.empty([len(img_files) - (time_step - 1), time_step], dtype='object')

    # If timestep == 1 then place the file names for the image in the list of images
    # else loop place the number of images indicated by the timestep in each row incrementing by
    #   one per row i.e. row 1 contains images 1-5, row 2 contains images 2-6, if the timestep was 5
    if time_step == 1:
        for i, img in enumerate(img_files):
            images[i] = img_files[i]
    else:
        for i, img in enumerate(img_files[:-(time_step-1)]):
            for j in range(time_step):
                images[i, j] = img_files[i+j]

    # return the list of imagepaths with the appropriate labels
    return images, label[time_step-1:]

# Same thing as the first one, just didn't feel like making the entire first one
#  in a if statement and have to pass a flag
def generate_video_frame_list_spherical(data_dir, model, movement, time_step):
    tracks = glob.glob(os.path.join(data_dir,'sphere' ,'*.csv'))
    tracks.sort()

    label = np.loadtxt(tracks[movement], delimiter=',')

    img_path = os.path.join(data_dir, models[model], '%02d' % movement)
    img_files = glob.glob(os.path.join(img_path, '*.png'))
    img_files.sort()

    images = np.empty([len(img_files) - (time_step - 1), time_step], dtype='object')

    if time_step == 1:
        for i, img in enumerate(img_files):
            images[i] = img_files[i]
    else:
        for i, img in enumerate(img_files[:-(time_step-1)]):
            for j in range(time_step):
                images[i, j] = img_files[i+j]
    label = np.expand_dims(label,1)
    return images, label[time_step-1:]

# Class to be instantiated in the main functions
class SynHead:
    # Initialization function, setup the fields
    def __init__(self, time_step, bin_size, max_angle):
        self.time_step = time_step
        self.bin_size = bin_size
        self.max_angle = max_angle

    # Creating the data given a data_dir, model_list, and movement_list
    def create_data(self, data_dir, model_list, move_list, ):
        # Create an empty matrix of 1xtimestep for the full list of images
        full_image_list = np.empty((0, self.time_step))
        # Create an empty matrix of 1x3 for the labels
        full_label_list = np.empty((0, 3))

        # Create a vector of index values where a new video starts
        video_start = np.empty([len(move_list), 1])

        # Create an empty list of backgrounds
        bg_index_crop = []
        # Concatinate *.jpg on to the path then separate
        bgfiles = glob.glob(os.path.join('../data/synhead2_release/background', '*.jpg'))
        bg_file_index = np.empty([len(move_list), 1])

        # For the movement and a model list generate a matrix of images and a matrix of labels
        for i, movement in enumerate(move_list):
            image_files, labels = generate_video_frame_list(data_dir,
                                                            model_list[i],
                                                            move_list[i],
                                                            self.time_step)
            # For each loop record the starting index of each new video
            if i != 0:
                video_start[i] = video_start[i-1] + image_files.shape[0]
            else:
                video_start[i] = image_files.shape[0]

            # Verticlely? Verticly? stack the the image files on the full image file list
            full_image_list = np.vstack((full_image_list, image_files))
            # Verticlely? Verticly? stack the the labels on the full labels list
            full_label_list = np.vstack((full_label_list, labels))

            # Randomly choose a background and load backgroundimage
            bg_index = np.random.randint(len(bgfiles))
            bgimg = cv2.imread(bgfiles[bg_index])

            # randomly Crop the images
            ix = np.random.randint(0, high=(bgimg.shape[1] - SYNHEAD_imW))
            iy = np.random.randint(0, high=(bgimg.shape[0] - SYNHEAD_imH))
            bgimg = bgimg[iy:iy + SYNHEAD_imH, ix:ix + SYNHEAD_imW, :]
            bg_index_crop.append(bgimg)

        # Originally was going to predict all three at once, instead I split them up
        #  Into individual labels
        full_pitch_labels = full_label_list[:, 0]
        full_yaw_labels = full_label_list[:, 1]
        full_roll_labels = full_label_list[:, 2]

        # Create bins from -max_angle to max_angle in accordance to bin size
        bins = np.linspace(-self.max_angle, self.max_angle, int((self.max_angle*2 + 1)/self.bin_size))
        self.num_bins = len(bins) + 1

        # Turn raw angles into bin numbers for labeling
        full_pitch_label_digitize = np.digitize(full_pitch_labels, bins)
        full_yaw_label_digitize = np.digitize(full_yaw_labels, bins)
        full_roll_label_digitize = np.digitize(full_roll_labels, bins)

        # Return everything
        return full_image_list, \
               full_pitch_label_digitize, full_yaw_label_digitize, full_roll_label_digitize, \
               bg_index_crop, video_start

# Basically the same thing, but generate_video_frame_list_spherical is called
# and the bins are already digitized so that portion is removed
class SynHeadSpherical:
    def __init__(self, time_step):
        self.time_step = time_step
        self.num_bins = 187

    def create_data(self, data_dir, model_list, move_list, ):
        full_image_list = np.empty((0, self.time_step))
        full_label_list = np.empty((0,1))
        video_start = np.empty([len(move_list), 1])
        bg_index_crop = []
        bgfiles = glob.glob(os.path.join('../data/synhead2_release/background', '*.jpg'))
        bg_file_index = np.empty([len(move_list), 1])
        for i, movement in enumerate(move_list):
            image_files, labels = generate_video_frame_list_spherical(data_dir,
                                                                       model_list[i],
                                                                       move_list[i],
                                                                       self.time_step)

            if i != 0:
                video_start[i] = video_start[i-1] + image_files.shape[0]
            else:
                video_start[i] = image_files.shape[0]

            full_image_list = np.vstack((full_image_list, image_files))
            full_label_list = np.vstack((full_label_list, labels))

            bg_index = np.random.randint(len(bgfiles))
            bgimg = cv2.imread(bgfiles[bg_index])

            ix = np.random.randint(0, high=(bgimg.shape[1] - SYNHEAD_imW))
            iy = np.random.randint(0, high=(bgimg.shape[0] - SYNHEAD_imH))
            bgimg = bgimg[iy:iy + SYNHEAD_imH, ix:ix + SYNHEAD_imW, :]
            bg_index_crop.append(bgimg)

        return full_image_list, full_label_list, \
               bg_index_crop, video_start
