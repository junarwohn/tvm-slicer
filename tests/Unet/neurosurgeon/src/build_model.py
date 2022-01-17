from SlicingMachine import TVMSlicer
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import numpy as np
import os
import json
import pygraphviz as pgv
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='type of devices [llvm, cuda]')
parser.add_argument('--partition_point', type=int, default=0, help='set partition point')
args = parser.parse_args()

np.random.seed(0)
img_size = 512
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model('./model/unet_{}.h5'.format(img_size))

## tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
target = 'cuda'
#target = 'llvm'
dev = tvm.cuda()
#dev = tvm.cpu()

with tvm.transform.PassContext(opt_level=2):
    lib = relay.build(mod, target, params=params)
    lib.export_library("./model/unet_tvm.so")

with open("./graph/graph_json_front.json", "r") as json_graph_front:
    json_graph_front = json.load(json_graph_front)
    del json_graph_front['extra']
    with tvm.transform.PassContext(opt_level=2):
        lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(json_graph_front))
    lib.export_library("./model/unet_tvm_front.so")

with open("./graph/graph_json_back.json", "r") as json_graph_back:
    json_graph_back = json.load(json_graph_back)
    del json_graph_back['extra']
    with tvm.transform.PassContext(opt_level=2):
        lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(json_graph_back))
    lib.export_library("./model/unet_tvm_back.so")

