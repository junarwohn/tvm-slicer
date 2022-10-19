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

def _conv_block_2(in_data, in_dim, out_dim, act_fn):
    out = _conv_block(in_data, in_dim, out_dim, act_fn)
    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same')(out)
    out = layers.BatchNormalization()(out)
    return out


def Model(in_dim, out_dim, num_filter, mutation=[0, 0, 0, 0]):
    act_fn = layers.LeakyReLU(alpha=0.2)

    # input_layer = layers.Input(shape=(512,512,in_dim,))
    input_layer = layers.Input(shape=(256,256,in_dim,))
    # mutate the Unet 1
    down_1 = _conv_block_2(input_layer, in_dim, num_filter, act_fn)
    model = keras.models.Model(inputs=input_layer, outputs=down_1)
    return model