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

def _conv_block(in_data, out_dim):
    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same')(in_data)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(alpha=0.2)(out)   
    return out

def _conv_trans_block(in_data, out_dim):
    out = layers.Conv2DTranspose(filters=out_dim, kernel_size=3, strides=2, padding='same', output_padding=1)(in_data)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU(alpha=0.2)(out)   
    return out

def _maxpool(in_data):
    pool = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(in_data)
    return pool

def _conv_block_2(in_data, out_dim):
    out = _conv_block(in_data=in_data, out_dim=out_dim)
    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same')(out)
    out = layers.BatchNormalization()(out)
    return out


def _mutate_reduction(down, num_filter):
    num_filter = int(num_filter / 2)
    pool = _maxpool(down)
    down = _conv_block(in_data=pool, out_dim=num_filter)
    return down, num_filter

def _mutate_expansion(trans, num_filter):
    trans = _conv_trans_block(in_data=trans, out_dim=num_filter)
    num_filter = num_filter * 2
    return trans, num_filter


def UNet(in_dim, out_dim, num_filter, mutation=[0, 0, 0, 0]):
    if len(mutation) != 4:
        print("Wrong input (eg. mutatation=[0, 0, 0, 0]")
    
    input_layer = layers.Input(shape=(256,256,in_dim,))


    ###########################
    ## Neural Reduction Block 0
    down_1 = _conv_block_2(in_data=input_layer, out_dim=num_filter)
    pool_1 = _maxpool(down_1)
    #- mutation-deep 0
    _num_filter = num_filter
    for i in range(mutation[0]):
        down_1, _num_filter = _mutate_reduction(down_1, _num_filter)
    #-
    ###########################
    


    ###########################
    ## Neural Reduction Block 1
    num_filter = num_filter*2
    down_2 = _conv_block_2(in_data=pool_1, out_dim=num_filter)
    pool_2 = _maxpool(down_2)
    #- mutation-deep 1
    _num_filter = num_filter
    for i in range(mutation[1]):
        down_2, _num_filter = _mutate_reduction(down_2, _num_filter)
    #-
    ###########################



    ###########################
    ## Neural Reduction Block 2
    num_filter = num_filter*2
    down_3 = _conv_block_2(in_data=pool_2, out_dim=num_filter)
    pool_3 = _maxpool(down_3)
    #- mutation-deep 2
    _num_filter = num_filter
    for i in range(mutation[2]):
        down_3, _num_filter = _mutate_reduction(down_3, _num_filter)
    #-
    ###########################



    ###########################
    ## Neural Reduction Block 3
    num_filter = num_filter*2
    down_4 = _conv_block_2(in_data=pool_3, out_dim=num_filter)
    pool_4 = _maxpool(down_4)
    #- mutation-deep 3
    _num_filter = num_filter
    for i in range(mutation[3]):
        down_4, _num_filter = _mutate_reduction(down_4, _num_filter)
    #-
    ###########################



    ###########################
    ## bridge
    num_filter = num_filter*2
    bridge = _conv_block_2(in_data=pool_4, out_dim=num_filter)
    ##
    ###########################



    ###########################
    ## Neural Expansion Block 3 
    num_filter = int(num_filter / 2)
    trans_1 = _conv_trans_block(in_data=bridge, out_dim=num_filter)
    
    #- mutation-shallow 3
    _num_filter = num_filter
    for i in range(mutation[3]):
        down_4, _num_filter = _mutate_expansion(down_4, _num_filter)
    #-

    concat_1 = tf.keras.layers.Concatenate(axis=3)([trans_1, down_4])
    up_1 = _conv_block_2(concat_1, out_dim=num_filter)
    ##
    ###########################



    ###########################
    ## Neural Expansion Block 2
    num_filter = int(num_filter / 2)
    trans_2 = _conv_trans_block(in_data=up_1, out_dim=num_filter)

    #- mutation-shallow 2
    _num_filter = num_filter
    for i in range(mutation[2]):
        down_3, _num_filter = _mutate_expansion(down_3, _num_filter)
    #-

    concat_2 = tf.keras.layers.Concatenate(axis=3)([trans_2, down_3])
    up_2 = _conv_block_2(in_data=concat_2, out_dim=num_filter)
    ##
    ###########################



    ###########################
    ## Neural Expansion Block 1
    num_filter = int(num_filter / 2)
    trans_3 = _conv_trans_block(in_data=up_2, out_dim=num_filter)
    
    #- mutation-shallow 1
    _num_filter = num_filter
    for i in range(mutation[1]):
        down_2, _num_filter = _mutate_expansion(down_2, _num_filter)
    #-
    
    concat_3 = tf.keras.layers.Concatenate(axis=3)([trans_3, down_2])
    up_3 = _conv_block_2(in_data=concat_3, out_dim=num_filter)
    ##
    ###########################



    ###########################
    ## Neural Expansion Block 0
    num_filter = int(num_filter / 2)
    trans_4 = _conv_trans_block(in_data=up_3, out_dim=num_filter)

    #- mutation-shallow 0
    _num_filter = num_filter
    for i in range(mutation[0]):
        down_1, _num_filter = _mutate_expansion(down_1, _num_filter)
    #-

    concat_4 = tf.keras.layers.Concatenate(axis=3)([trans_4, down_1])
    up_4 = _conv_block_2(in_data=concat_4, out_dim=num_filter)
    ##
    ###########################



    out = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same', activation=activations.sigmoid)(up_4)

    model = keras.models.Model(inputs=input_layer, outputs=out)
    return model