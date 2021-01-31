import pandas as pd
import numpy as np
import cv2
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle, class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def random_rotate(img, min_angle, max_angle):
    rotation_angle = random.uniform(min_angle, max_angle)
    rotated_img = ImageDataGenerator().apply_transform(x=img, transform_parameters={'theta': rotation_angle})
    return rotated_img


def horizontal_flip(img, ):
    return cv2.flip(img, 1)


class Dataset:
    def __init__(self, annotation_csv, data_path):
        self.data_path = data_path
        csv_data = pd.read_csv(annotation_csv)
        train, testval = train_test_split(csv_data, test_size=0.3, random_state=22)
        val, test = train_test_split(testval, test_size=0.5, random_state=23)

        self.train = train.reset_index(drop=True)
        self.val = val.reset_index(drop=True)
        self.test = test.reset_index(drop=True)

        one_hot_encoder = OneHotEncoder()

        train_y = np.array(train.status.tolist())
        self.train_y = one_hot_encoder.fit_transform(train_y.reshape(-1, 1)).toarray()

        val_y = np.array(val.status.tolist())
        self.val_y = one_hot_encoder.fit_transform(val_y.reshape(-1, 1)).toarray()

        test_y = np.array(test.status.tolist())
        self.test_y = one_hot_encoder.fit_transform(test_y.reshape(-1, 1)).toarray()

    def generator(self, data_type="train", batch_size=32, flip=False, rotate=False):
        if data_type is 'test':
            data = self.test
            y = self.test_y
        elif data_type is 'val':
            data = self.val
            y = self.val_y
        else:
            data = self.train
            y = self.train_y

        while True:
            for start in range(0, len(data), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(data))
                for i in range(start, end):
                    img = cv2.imread(os.path.join(self.data_path, data['filename'][i]))
                    img = cv2.resize(img, (224, 224))
                    x_batch.append(img)
                    y_batch.append(y[i])
                    if flip:
                        flip_img = horizontal_flip(img)
                        x_batch.append(flip_img)
                        y_batch.append(y[i])
                    if rotate:
                        rotate_img = random_rotate(img, -90.0, 90.0)
                        x_batch.append(rotate_img)
                        y_batch.append(y[i])
                x_batch, y_batch = shuffle(x_batch, y_batch)
                yield np.array(x_batch), np.array(y_batch)

    def get_class_weight(self):
        train_label_list = self.train['status'].tolist()
        weight = class_weight.compute_class_weight(
            'balanced',
            np.unique(train_label_list),
            train_label_list,
        )
        calculated_weights = {
            0: weight[0],
            1: weight[1],
        }
        return calculated_weights
