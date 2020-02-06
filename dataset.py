import numpy as np
import glob
import os
import cv2

data_sets = ['BIWI', 'breitenstein', 'kinect', 'softkinect']
models = ['f01', 'f02', 'f03', 'f04', 'f05', 'm01', 'm02', 'm03', 'm04', 'm05']
SYNHEAD_imW = 400
SYNHEAD_imH = 400

np.random.seed(42)

def generate_video_frame_list(data_dir, model, movement, time_step):
    tracks = glob.glob(os.path.join(data_dir, '*.csv'))
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

    return images, label[time_step-1:]

def generate_video_frame_list_sphereical(data_dir, model, movement, time_step):
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

#--- SynHead Dataset
class SynHead:
    def __init__(self, time_step, bin_size, max_angle):
        self.time_step = time_step
        self.bin_size = bin_size
        self.max_angle = max_angle

    def create_data(self, data_dir, model_list, move_list, ):
        full_image_list = np.empty((0, self.time_step))
        full_label_list = np.empty((0, 3))
        video_start = np.empty([len(move_list), 1])
        bg_index_crop = []
        bgfiles = glob.glob(os.path.join('../data/synhead2_release/background', '*.jpg'))
        bg_file_index = np.empty([len(move_list), 1])
        for i, movement in enumerate(move_list):
            image_files, labels = generate_video_frame_list(data_dir,
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

        full_pitch_labels = full_label_list[:, 0]
        full_yaw_labels = full_label_list[:, 1]
        full_roll_labels = full_label_list[:, 2]

        bins = np.linspace(-self.max_angle, self.max_angle, int((self.max_angle*2 + 1)/self.bin_size))
        self.num_bins = len(bins) + 1

        full_pitch_label_digitize = np.digitize(full_pitch_labels, bins)
        full_yaw_label_digitize = np.digitize(full_yaw_labels, bins)
        full_roll_label_digitize = np.digitize(full_roll_labels, bins)

        return full_image_list, \
               full_pitch_label_digitize, full_yaw_label_digitize, full_roll_label_digitize, \
               bg_index_crop, video_start

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
            image_files, labels = generate_video_frame_list_sphereical(data_dir,
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