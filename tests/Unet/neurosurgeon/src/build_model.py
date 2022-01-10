from SlicingMachine import TVMSlicer
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import numpy as np
import os
import json
import pygraphviz as pgv

np.random.seed(0)
input_data = np.random.normal(0,1,(1,512,512,3)).astype(np.float32)
model_keras = tf.keras.models.load_model('./model/unet.h5')

## tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
target = 'cuda'

dev = tvm.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

graph_json_raw = lib['get_graph_json']()
graph_json_front_info, graph_json_back_info = TVMSlicer(graph_json_raw, [[0,49],[49,111]]).get_graph()

graph_json_front, input_front, output_front = graph_json_front_info
graph_json_back, input_back, output_back = graph_json_back_info

