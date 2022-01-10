from SlicingMachine import TVMSlicer
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import numpy as np
import os
import json
import pygraphviz as pgv

os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

np.random.seed(0)
input_data = np.random.normal(0,1,(1,512,512,3)).astype(np.float32)

## import original and check output
# model_keras = tf.keras.models.load_model('./unet_512.h5')
model_keras = tf.keras.models.load_model('./unet_512_keras.h5')
# input_tensor = tf.convert_to_tensor(input_data)
out_keras = model_keras(input_data).numpy()
print(out_keras[0][0][:10].T)
print(out_keras.shape)

## tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)
target = 'cuda'

dev = tvm.cuda()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

model_tvm = graph_executor.GraphModule(lib["default"](dev))
model_tvm.set_input('input_1', input_data)
model_tvm.run()
out_tvm = model_tvm.get_output(0).asnumpy()

out_tvm = out_tvm.transpose([0, 2, 3, 1])
print(out_tvm[0][0][:10].T)
print(out_tvm.shape)
graph_json_raw = lib['get_graph_json']()
# graph_json_front, graph_json_back = TVMSlicer(lib['get_graph_json'](), 10).get_graph()
graph_json_front, graph_json_back = TVMSlicer(lib['get_graph_json'](), 29).get_graph()

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(graph_json_front))

model_tvm_front = graph_executor.GraphModule(lib["default"](dev))
model_tvm_front.set_input('input_1', input_data)
model_tvm_front.run()
out_tvm_front_1 = model_tvm_front.get_output(0).asnumpy()
out_tvm_front_2 = model_tvm_front.get_output(1).asnumpy()
##
out_tvm_front_3 = model_tvm_front.get_output(2).asnumpy()

print(out_tvm_front_1.shape, out_tvm_front_2.shape)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_graph(mod, target=target, target_host=None, params=params, mod_name="default", graph_config=json.dumps(graph_json_back))


model_tvm_back = graph_executor.GraphModule(lib["default"](dev))
model_tvm_back.set_input('input_1', out_tvm_front_1)
model_tvm_back.set_input('input_2', out_tvm_front_2)
model_tvm_back.set_input('input_3', out_tvm_front_3)
model_tvm_back.run()
out_tvm_back = model_tvm_back.get_output(0).asnumpy()

out_tvm_back = out_tvm_back.transpose([0, 2, 3, 1])
print(out_tvm_back.shape)
print(out_tvm_back[0][0][:10].T)

print(np.allclose(out_tvm, out_tvm_back))

def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]), node['name'] + '[{}]'.format(node_idx))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')

show_graph(graph_json_raw, file_name='raw')
show_graph(graph_json_front, file_name='front')
show_graph(graph_json_back, file_name='back')