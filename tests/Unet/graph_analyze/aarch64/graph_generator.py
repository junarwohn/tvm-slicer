import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import numpy as np
import os
import json
import pygraphviz as pgv

def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]), node['name'] + '[{}]'.format(node_idx))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')

os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

np.random.seed(12)
input_data = np.random.normal(0,1,(1,512,512,3)).astype(np.float32)

# Load keras model
# model_keras = tf.keras.models.load_model('./unet_512.h5')
model_keras = tf.keras.models.load_model('./unet_512_keras.h5')

input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)

target = 'cuda'
#target = 'llvm'

#dev = tvm.cuda()
dev = tvm.cpu()

for i in range(4):
    with tvm.transform.PassContext(opt_level=i):
        lib = relay.build(mod, target, params=params)

    show_graph(lib['get_graph_json'](), "unet_{}_lv_{}".format(target, i))
    with open("unet_{}_lv_{}.json".format(target, i), "w") as json_file:
        json_file.write(lib['get_graph_json']())
