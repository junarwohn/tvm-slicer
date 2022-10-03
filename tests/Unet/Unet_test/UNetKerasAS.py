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



def _conv_block(in_data, in_dim, out_dim, act_fn):
    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same')(in_data)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(alpha=0.2)(out)   
    return out

def _conv_trans_block(in_data, in_dim, out_dim, act_fn):
    out = layers.Conv2DTranspose(filters=out_dim, kernel_size=3, strides=2, padding='same', output_padding=1)(in_data)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(alpha=0.2)(out)   
    return out

def _conv_deep_trans_block(in_data, in_dim, out_dim, act_fn):
    out = layers.Conv2DTranspose(filters=out_dim, kernel_size=5, strides=4, padding='same', output_padding=3)(in_data)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(alpha=0.2)(out)   
    return out

def _maxpool(in_data):
    pool = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(in_data)
    return pool


def _deep_maxpool(in_data):
    pool = layers.MaxPool2D(pool_size=(4,4), strides=4, padding='same')(in_data)
    return pool

def _conv_block_2(in_data, in_dim, out_dim, act_fn):
    out = _conv_block(in_data, in_dim, out_dim, act_fn)
    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same')(out)
    out = layers.BatchNormalization()(out)
    return out




def UNet(in_dim, out_dim, num_filter, mutation=[0, 0, 0, 0]):
    if len(mutation) != 4:
        print("Wrong input (eg. mutatation=[0, 0, 0, 0]")

    act_fn = layers.LeakyReLU(alpha=0.2)

    # input_layer = layers.Input(shape=(512,512,in_dim,))
    input_layer = layers.Input(shape=(256,256,in_dim,))
    # mutate the Unet 1
    down_1 = _conv_block_2(input_layer, in_dim, num_filter, act_fn)
    for i in range(mutation[0]):
        num_filter = int(num_filter / 2)
        pool_1 = _maxpool(down_1)
        down_1 = _conv_block_2(pool_1, num_filter*2, num_filter, act_fn)
        # down_1 = _conv_block_2(pool_1, num_filter, num_filter*2, act_fn)
        # down_1 = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(down_1)
    pool_1 = _maxpool(down_1)
    # print("down_1", down_1.shape)


    # mutate the Unet 2
    down_2 = _conv_block_2(pool_1, num_filter*1, num_filter*2, act_fn)
    for i in range(mutation[1]):
        num_filter = int(num_filter / 2)
        pool_2 = _maxpool(down_2)
        down_2 = _conv_block_2(pool_2, num_filter*2, num_filter, act_fn)
        # down_2 = _conv_block_2(pool_2, num_filter, num_filter*2, act_fn)
        # down_2 = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(down_2)
    pool_2 = _maxpool(down_2)
    # print("down_2", down_2.shape)

    # mutate the Unet 3
    num_filter = num_filter * 2
    down_3 = _conv_block_2(pool_2, num_filter*1, num_filter*2, act_fn)
    for i in range(mutation[2]):
        num_filter = int(num_filter / 2)
        pool_3 = _maxpool(down_3)
        down_3 = _conv_block_2(pool_3, num_filter*2, num_filter, act_fn)
        # down_3 = _conv_block_2(pool_3, num_filter, num_filter*2, act_fn)
        # down_3 = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(down_3)
    pool_3 = _maxpool(down_3)
    # print("down_3", down_3.shape)

    # mutate the Unet 4
    num_filter = num_filter * 2
    down_4 = _conv_block_2(pool_3, num_filter*1, num_filter*2, act_fn)
    for i in range(mutation[3]):
        num_filter = int(num_filter / 2)
        pool_4 = _maxpool(down_4)
        down_4 = _conv_block_2(pool_4, num_filter*2, num_filter, act_fn)
        # down_4 = _conv_block_2(pool_4, num_filter, num_filter*2, act_fn)
        # down_4 = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(down_4)
    pool_4 = _maxpool(down_4)
    # print("down_4", down_4.shape)

    num_filter = num_filter * 2
    bridge = _conv_block_2(pool_4, num_filter*1, num_filter*2, act_fn)
    # print("bridge", bridge.shape)
    # mutate the Unet 4
    trans_1 = _conv_trans_block(bridge, num_filter*2, num_filter*1, act_fn)

    concat_1 = tf.keras.layers.Concatenate(axis=3)([trans_1, down_4])
    # print("concat_1", concat_1.shape)
    up_1 = _conv_block_2(concat_1, num_filter*2, num_filter*1, act_fn)
    # print("up_1", up_1.shape)


    # mutate the Unet 4
    num_filter = int(num_filter / 2)
    trans_2 = _conv_trans_block(up_1, num_filter*2, num_filter*1, act_fn)
    for i in range(mutation[3]):
        num_filter = num_filter * 2
        trans_2 = _conv_trans_block(trans_2, num_filter*2, num_filter*1, act_fn)


    concat_2 = tf.keras.layers.Concatenate(axis=3)([trans_2, down_3])
    # print("concat_2", concat_2.shape)
    up_2 = _conv_block_2(concat_2, num_filter*2, num_filter*1, act_fn)
    # print("up_2", up_2.shape)

    # mutate the Unet 3
    num_filter = int(num_filter / 2)
    trans_3 = _conv_trans_block(up_2, num_filter*2, num_filter*1, act_fn)
    for i in range(mutation[2]):
        num_filter = num_filter * 2
        trans_3 = _conv_trans_block(trans_3, num_filter*2, num_filter*1, act_fn)

    
    concat_3 = tf.keras.layers.Concatenate(axis=3)([trans_3, down_2])
    # print("concat_3", concat_3.shape)
    up_3 = _conv_block_2(concat_3, num_filter*2, num_filter*1, act_fn)
    # print("up_3", up_3.shape)

    # mutate the Unet 2
    num_filter = int(num_filter / 2)
    trans_4 = _conv_trans_block(up_3, num_filter*2, num_filter*1, act_fn)
    # print("trans_4", trans_4.shape)
    for i in range(mutation[1]):
        num_filter = num_filter * 2
        trans_4 = _conv_trans_block(trans_4, num_filter*2, num_filter*1, act_fn)

        # print("trans_4", trans_4.shape)

    concat_4 = tf.keras.layers.Concatenate(axis=3)([trans_4, down_1])
    # print("concat_4", concat_4.shape)

    up_4 = _conv_block_2(concat_4, num_filter*2, num_filter*1, act_fn)
    # print("up_4", up_4.shape)

    for i in range(mutation[0]):
        num_filter = num_filter * 2
        up_4 = _conv_trans_block(up_4, num_filter*2, num_filter*1, act_fn)

        # print("up_4", up_4.shape)

    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(up_4)
    # print("out", out.shape)

    model = keras.models.Model(inputs=input_layer, outputs=out)
    return model