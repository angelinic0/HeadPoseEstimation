import tensorflow as tf
import cv2
import numpy as np


def compose_image(img, bgimg):
    im    = img[:, :, :3]
    alpha = img[:, :, 2]
    alpha = (alpha > 0).astype(np.int32)
    imag  = np.array(bgimg)
    for c in range(3):
        imag[:, :, c] = im[:, :, c] * alpha + bgimg[:, :, c] * (1 - alpha)
    return imag

class SynHeadSequencer(tf.keras.utils.Sequence):
    def __init__(self, X_data, y_data, background, video_start, batch_size, time_steps, input_shape, num_classes, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.background = background
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.video_start = video_start

        self.shuffle = shuffle
        # Array of indexes with shuffle
        if self.shuffle:
            self.indexes = np.arange(len(self.X_data))
            np.random.shuffle(self.indexes)
        else:
            self.indexes = np.arange(len(self.X_data))

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.time_steps == 1:
            X = np.ndarray((self.batch_size, *self.input_shape), dtype=float)
            y = np.empty((self.batch_size, 1), dtype=float)

            for i, idx in enumerate(indexes):
                video = 0
                while index > self.video_start[video]:
                    video += 1
                bgimg = self.background[video]

                frame = cv2.imread(self.X_data[idx][0], 1)
                comp_image = compose_image(frame, bgimg)
                res_img = np.asarray(cv2.resize(comp_image, (self.input_shape[0], self.input_shape[1])))
                normed_img = (res_img - res_img.mean()) / res_img.std()
                X[i, :] = normed_img
                y[i] = self.y_data[idx]

            return X, tf.keras.utils.to_categorical(y, self.num_classes)
        else:
            X = np.ndarray((self.batch_size, self.time_steps, *self.input_shape), dtype=float)
            y = np.empty((self.batch_size, 1), dtype=float)

            for i, index in enumerate(indexes):
                video = 0
                while index > self.video_start[video]:
                    video += 1
                bgimg = self.background[video]

                for time_idx in range(self.time_steps):
                    frame = cv2.imread(self.X_data[index][time_idx], 1)
                    comp_image = compose_image(frame, bgimg)
                    res_img = np.asarray(cv2.resize(comp_image, (self.input_shape[0], self.input_shape[1])))
                    normed_img = (res_img - res_img.mean()) / res_img.std()
                    X[i, time_idx, :] = normed_img
                y[i] = self.y_data[index]
            return X, tf.keras.utils.to_categorical(y, self.num_classes)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)