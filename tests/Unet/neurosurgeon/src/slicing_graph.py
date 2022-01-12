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
#img_size = 128
img_size = 512 
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
model_keras = tf.keras.models.load_model('./model/unet_{}.h5'.format(img_size))

## tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
#target = 'cuda'
target = 'llvm'

#dev = tvm.cuda()
dev = tvm.cpu()
with tvm.transform.PassContext(opt_level=2):
    lib = relay.build(mod, target, params=params)

graph_json_raw = lib['get_graph_json']()
graph_json_front_info, graph_json_back_info = TVMSlicer(graph_json_raw, [[0,10],[10,121]]).get_graph()
# graph_json_front_info, graph_json_back_info = TVMSlicer(graph_json_raw, [[0,9],[9,111]]).get_graph()

graph_json_front, input_front, output_front = graph_json_front_info
graph_json_back, input_back, output_back = graph_json_back_info

with open("./graph/graph_json_front.json", "w") as json_file:
    graph_json_front['extra'] = {}
    graph_json_front['extra']['inputs'] = input_front
    graph_json_front['extra']['outputs'] = output_front
    json_file.write(json.dumps(graph_json_front))


with open("./graph/graph_json_back.json", "w") as json_file:
    graph_json_back['extra'] = {}
    graph_json_back['extra']['inputs'] = output_front
    graph_json_back['extra']['outputs'] = output_back
    json_file.write(json.dumps(graph_json_back))
