
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def UNet(in_dim, out_dim, num_filter):
    # act_fn = layers.LeakyReLU(alpha=0.2)

    input_layer = layers.Input(shape=(512,512,3))
    down_1 = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(input_layer)
    pool_1 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(down_1)
    bridge = layers.Conv2D(filters=num_filter*2, kernel_size=3, strides=1, padding='same')(pool_1)
    trans_1 = layers.Conv2DTranspose(filters=num_filter, kernel_size=3, strides=2, padding='same', output_padding=1)(bridge)
    concat_1 = tf.keras.layers.Concatenate(axis=3,)([trans_1, down_1])
    up_1 = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(concat_1)
    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(up_1)
    # model = keras.models.Model(inputs=input_layer, outputs=[out, bridge])
    model = keras.models.Model(inputs=input_layer, outputs=out)

    return model
"""
    input_layer = layers.Input(shape=(512,512,3))
    out = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(input_layer)
    # out = layers.BatchNormalization()(out)
    # out = layers.LeakyReLU(alpha=0.2)(out)

    down_1 = out

    pool_1 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(down_1)

    out = layers.Conv2D(filters=num_filter*2, kernel_size=3, strides=1, padding='same')(pool_1)
    # out = layers.BatchNormalization()(out)
    # out = layers.LeakyReLU(alpha=0.2)(out)

    down_2 = out

    pool_2 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(down_2)

    bridge = layers.Conv2D(filters=num_filter*4, kernel_size=3, strides=1, padding='same')(pool_2)

    trans_1 = layers.Conv2DTranspose(filters=num_filter*2, kernel_size=3, strides=2, padding='same', output_padding=1)(bridge)

    concat_1 = tf.keras.layers.Concatenate(axis=3,)([trans_1, down_2])

    up_1 = layers.Conv2D(filters=num_filter*2, kernel_size=3, strides=1, padding='same')(concat_1)

    trans_2 = layers.Conv2DTranspose(filters=num_filter, kernel_size=3, strides=2, padding='same', output_padding=1)(up_1)

    concat_2 = tf.keras.layers.Concatenate(axis=3,)([trans_2, down_1])

    up_2 = layers.Conv2D(filters=num_filter, kernel_size=3, strides=1, padding='same')(concat_2)

    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(up_2)

    model = keras.models.Model(inputs=input_layer, outputs=out)
"""
