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

with open("./graph/graph_json_front.json", "r") as json_graph_front:
    json_graph_front = json.load(json_graph_front)
    del json_graph_front['extra']
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(json_graph_front))
    lib.export_library("./model/unet_tvm_front.so")

with open("./graph/graph_json_back.json", "r") as json_graph_back:
    json_graph_back = json.load(json_graph_back)
    del json_graph_back['extra']
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(json_graph_back))
    lib.export_library("./model/unet_tvm_back.so")

