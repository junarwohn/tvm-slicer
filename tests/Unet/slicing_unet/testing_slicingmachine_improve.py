from SlicingMachine_improve import TVMSlicer
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
model_keras = tf.keras.models.load_model('./unet_512.h5')

## tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
target = 'cuda'

dev = tvm.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

graph_json_raw = lib['get_graph_json']()
# graph_json_front, graph_json_back = TVMSlicer(lib['get_graph_json'](), 10).get_graph()
print(TVMSlicer(lib['get_graph_json'](), [[7,11]]).get_graph())
