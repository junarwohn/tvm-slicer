from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def VerySimpleModel():
    input_layer = layers.Input(shape=(256,256,3))
    out = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(input_layer)
    out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
    out = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(out)
    out2 = out
    out2 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out2)
    out2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(out2)
    out2 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out2)
    model = keras.models.Model(inputs=input_layer, outputs=out2)
    return model


    # out = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
    # out = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
    # out = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
    # out = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
