from UNetKerasMod import UNet
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import time
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

batch_size = 4
img_size = (512, 512)
in_dim = 3
out_dim = 1
num_filters = 16
epochs=100


def DataGenerator(data_gen_args, path, random_seed, target_size, batch_size, color_mode='rgb', subset='training'):
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    data_gen = image_gen.flow_from_directory(path, class_mode=None, target_size=target_size, color_mode=color_mode, shuffle=True, seed=random_seed, batch_size=batch_size, subset=subset)
    return data_gen


# y = DataGenerator(path, random_seed=1, is_categorical=True, threshold=250, color_mode=0)
# data_gen_args = dict(
#     rescale=1./255,
#         #                     featurewise_center=True,
#         #                     featurewise_std_normalization=True,
#         #                 rotation_range=30,
#         #                 width_shift_range=0.1,
#         #                 height_shift_range=0.1,
#         #                 zoom_range=0.2)
# )
data_gen_args = dict(
    rescale=1./255,
)
# dataset_type = ['train', 'validation', 'test']
raw_data_path = "./data/us-bmod/{}/raw"
mask_data_path = "./data/us-bmod/{}/mask"

x_train = DataGenerator(data_gen_args, raw_data_path.format('train'), random_seed=1, target_size=img_size, batch_size=batch_size)
x_validation = DataGenerator(data_gen_args, raw_data_path.format('validation'), random_seed=1, target_size=img_size, batch_size=batch_size)
x_test = DataGenerator(data_gen_args, raw_data_path.format('test'), random_seed=1, target_size=img_size, batch_size=batch_size)

data_gen_args = dict(
    rescale=1./255,
    preprocessing_function = lambda x: np.where(x>10, 255, 0).astype(x.dtype)
)
y_train = DataGenerator(data_gen_args, mask_data_path.format('train'), random_seed=1, target_size=img_size, color_mode='grayscale', batch_size=batch_size)
y_validation = DataGenerator(data_gen_args, mask_data_path.format('validation'), random_seed=1, target_size=img_size, color_mode='grayscale', batch_size=batch_size)
y_test = DataGenerator(data_gen_args, mask_data_path.format('test'), random_seed=1, target_size=img_size, color_mode='grayscale', batch_size=batch_size)

data_generator_train = zip(x_train, y_train)
data_generator_validation = zip(x_validation, y_validation)
data_generator_test = zip(x_test, y_test)

for i in range(0, 5):
    model_file_name = "./checkpoint-epoch-{epoch:04d}-" + time.strftime("%y%m%d-%H%M") + "maxpool_{}th-compression".format(i) + "-512.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_file_name,
        monitor='binary_crossentropy',  
        verbose=1,            # 로그를 출력합니다
        save_best_only=True,  # 가장 best 값만 저장합니다
    #     save_weight_only=True,
        mode='auto'
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=5)
    if i == 0:
        model = UNet(3,1,16, [])
    else:
        model = UNet(3,1,16, [i])
    batch_size = 4
    img_size = (512,512)
    model.build(input_shape=(batch_size,img_size,3))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['binary_crossentropy'])
    model.fit(data_generator_train, epochs=50, steps_per_epoch=len(x_train)-1, callbacks=[checkpoint, early_stop], validation_data=data_generator_validation, validation_steps=50)
