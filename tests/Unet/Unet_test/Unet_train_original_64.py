from UNetKerasAS import UNet as UnetAS
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
import random
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

batch_size = 4
img_size = (512, 512)
in_dim = 3
out_dim = 1
num_filters = 64
epochs=100
# random_seed = random.randint(0, 32)
random_seed = 0

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
base_path = os.environ.get("TS_DATA_PATH", "")
raw_data_path = base_path + "/us-bmod/{}/raw"
mask_data_path = base_path + "/us-bmod/{}/mask"

x_train = DataGenerator(data_gen_args, raw_data_path.format('train'), random_seed=random_seed, target_size=img_size, batch_size=batch_size)
x_validation = DataGenerator(data_gen_args, raw_data_path.format('validation'), random_seed=random_seed, target_size=img_size, batch_size=batch_size)
# x_test = DataGenerator(data_gen_args, raw_data_path.format('test'), random_seed=random_seed, target_size=img_size, batch_size=batch_size)

data_gen_args = dict(
    rescale=1./255,
    preprocessing_function = lambda x: np.where(x>10, 255, 0).astype(x.dtype)
)
y_train = DataGenerator(data_gen_args, mask_data_path.format('train'), random_seed=random_seed, target_size=img_size, color_mode='grayscale', batch_size=batch_size)
y_validation = DataGenerator(data_gen_args, mask_data_path.format('validation'), random_seed=random_seed, target_size=img_size, color_mode='grayscale', batch_size=batch_size)
# y_test = DataGenerator(data_gen_args, mask_data_path.format('test'), random_seed=random_seed, target_size=img_size, color_mode='grayscale', batch_size=batch_size)

data_generator_train = zip(x_train, y_train)
data_generator_validation = zip(x_validation, y_validation)
# data_generator_test = zip(x_test, y_test)

def combination_gen(n, k, combi, ret):
    if n == 0:
        for i in range(k):
            combi.append(0)
        ret.append(copy.deepcopy(combi))
    elif k == 0:
        ret.append(copy.deepcopy(combi))
    else:
        for i in range(n+1):
            combination_gen(n - i, k - 1, combi + [i], ret)
combinations = []
total_limit = 5
depth = 4
combination_gen(5, 4, [], combinations)
# for i in range(0, 5):
for com in combinations:
    # model_file_name = "./checkpoint-epoch-{epoch:04d}-" + time.strftime("%y%m%d-%H%M") + "maxpool_{}th-compression".format(i) + "-512.h5"
    print("############################")
    print(com)
    print("############################")
    x_train.reset()
    y_train.reset()
    x_validation.reset()
    y_validation.reset()

    data_generator_train = zip(x_train, y_train)
    data_generator_validation = zip(x_validation, y_validation)

    model_file_name = "./model_{}th_512.h5".format(0)
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_file_name,
        monitor='binary_crossentropy',  
        verbose=1,            # 로그를 출력합니다
        save_best_only=True,  # 가장 best 값만 저장합니다
    #     save_weight_only=True,
        mode='auto'
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=5)
    # if i == 0:
    #     model = UNet(3,1,16, [>])
    # else:
    #     model = UNet(3,1,16, [i])
    model = UnetAS(3, 1, 16, com)
    batch_size = 4
    img_size = (512,512)
    model.build(input_shape=(batch_size,img_size,3))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['binary_crossentropy'])
    # model.fit(data_generator_train, epochs=50, steps_per_epoch=len(x_train)-1, callbacks=[checkpoint, early_stop], validation_data=data_generator_validation, validation_steps=50)
    model.fit(data_generator_train, epochs=10, steps_per_epoch=len(x_train)-1, callbacks=[early_stop], validation_data=data_generator_validation, validation_steps=10, verbose=2)
