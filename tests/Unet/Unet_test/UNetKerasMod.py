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




def UNet(in_dim, out_dim, num_filter, maxpool_mutate=[]):
    act_fn = layers.LeakyReLU(alpha=0.2)

    # input_layer = layers.Input(shape=(128,128,in_dim,))
    input_layer = layers.Input(shape=(512,512,in_dim,))
    down_1 = _conv_block_2(input_layer, in_dim, num_filter, act_fn)

    if 1 not in maxpool_mutate:
        pool_1 = _maxpool(down_1)
    else:
        pool_1 = _deep_maxpool(down_1)

    down_2 = _conv_block_2(pool_1, num_filter*1, num_filter*2, act_fn)
    
    if 2 not in maxpool_mutate:
        pool_2 = _maxpool(down_2)
    else:
        pool_2 = _deep_maxpool(down_2)
    
    down_3 = _conv_block_2(pool_2, num_filter*2, num_filter*4, act_fn)

    if 3 not in maxpool_mutate:
        pool_3 = _maxpool(down_3)
    else:
        pool_3 = _deep_maxpool(down_3)

    down_4 = _conv_block_2(pool_3, num_filter*4, num_filter*8, act_fn)

    if 4 not in maxpool_mutate:
        pool_4 = _maxpool(down_4)
    else:
        pool_4 = _deep_maxpool(down_4)

    bridge = _conv_block_2(pool_4, num_filter*8, num_filter*16, act_fn)

    if 4 not in maxpool_mutate:
        trans_1 = _conv_trans_block(bridge, num_filter*16, num_filter*8, act_fn)
    else:
        trans_1 = _conv_deep_trans_block(bridge, num_filter*16, num_filter*8, act_fn)

    concat_1 = tf.keras.layers.Concatenate(axis=3)([trans_1, down_4])
    up_1 = _conv_block_2(concat_1, num_filter*16, num_filter*8, act_fn)

    if 3 not in maxpool_mutate:
        trans_2 = _conv_trans_block(up_1, num_filter*8, num_filter*4, act_fn)
    else:
        trans_2 = _conv_deep_trans_block(up_1, num_filter*8, num_filter*4, act_fn)

    concat_2 = tf.keras.layers.Concatenate(axis=3)([trans_2, down_3])
    up_2 = _conv_block_2(concat_2, num_filter*8, num_filter*4, act_fn)

    if 2 not in maxpool_mutate:
        trans_3 = _conv_trans_block(up_2, num_filter*4, num_filter*2, act_fn)
    else:
        trans_3 = _conv_deep_trans_block(up_2, num_filter*4, num_filter*2, act_fn)
    
    concat_3 = tf.keras.layers.Concatenate(axis=3)([trans_3, down_2])
    up_3 = _conv_block_2(concat_3, num_filter*4, num_filter*2, act_fn)
    
    
    if 1 not in maxpool_mutate:
        trans_4 = _conv_trans_block(up_3, num_filter*2, num_filter*1, act_fn)
    else:
        trans_4 = _conv_deep_trans_block(up_3, num_filter*2, num_filter*1, act_fn)
    
    concat_4 = tf.keras.layers.Concatenate(axis=3)([trans_4, down_1])
    up_4 = _conv_block_2(concat_4, num_filter*2, num_filter*1, act_fn)

    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(up_4)
    # out = keras.Sequential(
    #     [
    #         layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid),
    #     ]
    # )(up_4)
        

    model = keras.models.Model(inputs=input_layer, outputs=out)
    return model